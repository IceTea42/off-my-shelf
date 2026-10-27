"""S1 core pipeline: photos or videos in, a validated ``ListingPack`` on disk out.

For each still: Muse Spark lists the items (noun phrases) -> SAM 3.1 segments each phrase ->
overlapping detections are merged -> each item crop gets a full Muse Spark listing.
"""

import asyncio
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAIError

from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.config import Settings
from off_my_shelf.imaging import PixelBox, crop, encode, mask_image, pixel_lock, studio_canvas
from off_my_shelf.media import Still, collect_inputs, stills
from off_my_shelf.meta_client import make_client, make_open_weights_client
from off_my_shelf.mocks import save_pack
from off_my_shelf.product_shots import EmptyImageResponseError, ProductPhotographer
from off_my_shelf.sam import SamSegmenter, Segment
from off_my_shelf.schema import (
    Detection,
    Item,
    ItemAnalysis,
    Language,
    ListingPack,
    ModelCall,
    SourceRef,
)
from off_my_shelf.spark import SparkAnalyzer, SparkReplyError

DUPLICATE_IOU = 0.6
SHOT_ATTEMPTS = 2
HIDDEN_MIN_SHARE = 0.02  # below this, tiny mask gaps are noise, not occlusion
MAX_DRIFT = 0.12  # mean visible-pixel change above which a completion redrew the item
CROP_PADDING = 0.08
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Services:
    sam: SamSegmenter
    spark: SparkAnalyzer
    photographer: ProductPhotographer | None = None


def build_services(
    settings: Settings, offline: bool = False, product_shots: bool = True
) -> Services:
    """Live services, or cache-only ones when ``offline`` (no key, no network, no cost).

    With ``use_spark_open``, item analysis runs on Muse Spark Open at the partner endpoint;
    segmentation and product shots stay on the Meta Model API.
    """
    cache = DiskCache(settings.cache_dir)
    meta = None if offline else make_client(settings)
    if settings.use_spark_open:
        spark_client = None if offline else make_open_weights_client(settings)
        spark_model, effort = settings.spark_open_model, None
    else:
        spark_client, spark_model = meta, settings.spark_model
        effort = settings.reasoning_effort
    photographer = ProductPhotographer(meta, settings.image_model, cache) if product_shots else None
    return Services(
        sam=SamSegmenter(meta, settings.sam_model, cache),
        spark=SparkAnalyzer(spark_client, spark_model, cache, settings.max_output_tokens, effort),
        photographer=photographer,
    )


@dataclass(frozen=True, slots=True)
class _Found:
    still: Still
    media_path: str
    segment: Segment
    calls: tuple[ModelCall, ...]
    slug: str


def _best(segments: Sequence[Segment]) -> Segment | None:
    """The most confident segment for a phrase (largest when confidence is missing)."""
    if not segments:
        return None
    return max(segments, key=lambda s: (s.confidence or 0.0, s.box.area))


def _merge_duplicates(found: list[_Found]) -> list[_Found]:
    """Drop detections that overlap a more confident one on the same still."""
    ordered = sorted(found, key=lambda f: f.segment.confidence or 0.0, reverse=True)
    kept: list[_Found] = []
    for candidate in ordered:
        if not any(
            k.still is candidate.still and k.segment.box.iou(candidate.segment.box) > DUPLICATE_IOU
            for k in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda f: (f.media_path, f.segment.box.left))


def _same_item(a: ItemAnalysis, b: ItemAnalysis) -> bool:
    ia, ib = a.identification, b.identification
    return (ia.name.casefold(), (ia.brand or "").casefold()) == (
        ib.name.casefold(),
        (ib.brand or "").casefold(),
    )


def slugify(text: str, max_words: int = 2) -> str:
    """'Cake Stand!' -> 'cake-stand': the last ``max_words`` ASCII words (English noun phrases
    end with the noun), accents folded, never empty."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_text.casefold())[-max_words:]
    return "-".join(words) or "item"


def _split_cost(call: ModelCall, parts: int) -> ModelCall:
    """One share of a per-still call, so per-item costs add up to the real total."""
    if parts <= 1 or call.cost_usd is None:
        return call
    return call.model_copy(update={"cost_usd": call.cost_usd / parts})


class Pipeline:
    def __init__(
        self, services: Services, settings: Settings, progress: Progress | None = None
    ) -> None:
        self.services = services
        self.settings = settings
        self.progress = progress or (lambda _message: None)
        self._limit = asyncio.Semaphore(settings.max_concurrency)
        self.items_total = 0
        self.items_done = 0

    async def run(
        self, inputs: Sequence[Path], out_dir: Path, language: Language | None = None
    ) -> ListingPack:
        language = language or self.settings.default_language
        files = collect_inputs(inputs)
        if not files:
            raise ValueError("no supported photos or videos in the inputs")
        (out_dir / "media").mkdir(parents=True, exist_ok=True)

        media_paths: list[str] = []
        found: list[_Found] = []
        for file in files:
            self.progress(f"reading {file.name}")
            for still in stills(file, self.settings.max_image_side_px):
                media_path = f"media/{still.stem}.jpg"
                (out_dir / media_path).write_bytes(encode(still.image, "JPEG"))
                media_paths.append(media_path)
                found.extend(await self._detect(still, media_path))

        detections = _merge_duplicates(found)
        self.items_total = len(detections)
        self.progress(f"writing listings for {len(detections)} items")
        items = await asyncio.gather(
            *(
                self._item(f"item-{n:03d}", f, out_dir, language)
                for n, f in enumerate(detections, start=1)
            )
        )
        pack = ListingPack(
            run_id=f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}",
            created_at=datetime.now(UTC),
            media=media_paths,
            items=_drop_video_repeats(list(items)),
        )
        save_pack(out_dir, pack)
        self.progress(f"done: {len(pack.items)} items")
        return pack

    async def _detect(self, still: Still, media_path: str) -> list[_Found]:
        inventory = await self.services.spark.inventory(still.image)
        unique = {item.phrase: item for item in inventory.value.items}
        self.progress(f"{media_path}: looking for {', '.join(unique) or 'nothing'}")

        async def segment(phrase: str) -> tuple[Segment | None, ModelCall]:
            async with self._limit:
                run = await self.services.sam.segment(still.image, phrase)
            return _best(run.segments), run.call

        results = await asyncio.gather(*(segment(p) for p in unique))
        hits = [
            (seg, call, slugify(item.short_name or item.phrase))
            for item, (seg, call) in zip(unique.values(), results, strict=True)
            if seg is not None
        ]
        share = _split_cost(inventory.call, len(hits))
        return [_Found(still, media_path, seg, (share, call), slug) for seg, call, slug in hits]

    async def _item(self, item_id: str, found: _Found, out_dir: Path, language: Language) -> Item:
        image = found.still.image
        box = _crop_box(found)
        item_crop = crop(image, box)
        stem = f"{item_id}-{found.slug}"
        crop_path, mask_path = f"crops/{stem}.jpg", f"masks/{stem}.png"
        for rel, data in (
            (crop_path, encode(item_crop, "JPEG")),
            (mask_path, encode(mask_image(found.segment.mask), "PNG")),
        ):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(data)

        async with self._limit:
            analysis = await self.services.spark.analyze(item_crop, language)
        shots, shot_calls = await self._product_shot(stem, found, analysis.value, out_dir)
        self.items_done += 1
        return Item(
            item_id=item_id,
            detection=Detection(
                source=SourceRef(
                    media=found.media_path,
                    frame_index=found.still.frame_index,
                    timestamp_s=found.still.timestamp_s,
                ),
                bbox=found.segment.box.normalised(image.width, image.height),
                score=found.segment.confidence if found.segment.confidence is not None else 1.0,
                crop_path=crop_path,
                mask_path=mask_path,
            ),
            analysis=analysis.value,
            product_shots=shots,
            calls=[*found.calls, analysis.call, *shot_calls],
        )

    async def _product_shot(
        self, stem: str, found: _Found, analysis: ItemAnalysis, out_dir: Path
    ) -> tuple[list[str], list[ModelCall]]:
        """Shots built from the photo's real pixels; hidden parts AI-completed when trusted.

        Always: a studio shot of the real item (no generation). When part of the item is
        hidden: Muse Image fills only the hidden areas, every visible pixel is restored from the
        photo, and the result is kept only if the model did not move or redraw the item (drift)
        and Muse Spark sees no extra object or invented damage. One corrected retry.
        """
        canvas = studio_canvas(found.still.image, found.segment.mask, _crop_box(found))
        studio = _save(out_dir, f"shots/{stem}.jpg", encode(canvas.real, "JPEG", quality=92))
        photographer = self.services.photographer
        if photographer is None or canvas.hidden_share <= HIDDEN_MIN_SHARE:
            return [studio], []
        _save(out_dir, f"inputs/{stem}-to-fill.png", encode(canvas.to_fill, "PNG"))
        name, defects = analysis.identification.name, analysis.condition.defects
        calls: list[ModelCall] = []
        rejected: str | None = None
        try:
            for _ in range(SHOT_ATTEMPTS):
                async with self._limit:
                    generated, call = await photographer.complete(
                        canvas.to_fill, name, defects, rejected
                    )
                calls.append(call)
                locked, drift = pixel_lock(generated, canvas)
                if drift > MAX_DRIFT:
                    rejected = f"the item was moved or redrawn (drift {drift:.2f})"
                    continue
                async with self._limit:
                    check = await self.services.spark.check_shot(canvas.real, locked, name, defects)
                calls.append(check.call)
                if check.value.passed:
                    self.progress(f"{stem}: hidden parts completed (drift {drift:.3f})")
                    jpeg = encode(locked, "JPEG", quality=92)
                    return [_save(out_dir, f"shots/{stem}-completed.jpg", jpeg), studio], calls
                rejected = check.value.reason
        except (
            OpenAIError,
            EmptyImageResponseError,
            OfflineCacheMissError,
            SparkReplyError,
        ) as error:
            rejected = type(error).__name__
        self.progress(f"{stem}: real-pixel shot only (completion rejected: {rejected})")
        return [studio], calls


def _save(out_dir: Path, rel: str, data: bytes) -> str:
    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / rel).write_bytes(data)
    return rel


def _crop_box(found: _Found) -> PixelBox:
    image = found.still.image
    return found.segment.box.padded(CROP_PADDING, image.width, image.height)


def _drop_video_repeats(items: list[Item]) -> list[Item]:
    """A video shows the same item in several frames: keep its most confident sighting."""
    kept: list[Item] = []
    for item in sorted(items, key=lambda i: i.analysis.identification.confidence, reverse=True):
        repeat = item.detection.source.frame_index is not None and any(
            k.detection.source.frame_index is not None and _same_item(k.analysis, item.analysis)
            for k in kept
        )
        if not repeat:
            kept.append(item)
    return sorted(kept, key=lambda i: i.item_id)
