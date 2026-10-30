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
from off_my_shelf.imaging import (
    PixelBox,
    StudioCanvas,
    crop,
    encode,
    mask_image,
    pixel_lock,
    studio_canvas,
)
from off_my_shelf.media import Still, collect_inputs, stills
from off_my_shelf.meta_client import make_client, make_open_weights_client, open_weight_endpoints
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
from off_my_shelf.spark import ListingChain, ListingProvider, SparkAnalyzer, SparkReplyError

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
    listing: ListingChain
    photographer: ProductPhotographer | None = None


def build_services(
    settings: Settings, offline: bool = False, product_shots: bool = True
) -> Services:
    """Live services, or cache-only ones when ``offline`` (no key, no network, no cost).

    Listings run on Muse Spark Open at each configured partner (Baseten, then Modal), with
    Muse Spark on the Meta Model API as the last fallback; every other step uses the Meta API.
    """
    cache = DiskCache(settings.cache_dir)
    meta = None if offline else make_client(settings)
    spark = SparkAnalyzer(
        meta, settings.spark_model, cache, settings.max_output_tokens, settings.reasoning_effort
    )
    partners = [
        ListingProvider(
            name,
            SparkAnalyzer(
                None if offline else make_open_weights_client(settings, url, key),
                settings.spark_open_model,
                cache,
                settings.max_output_tokens,
            ),
        )
        for name, url, key in open_weight_endpoints(settings)
    ]
    photographer = ProductPhotographer(meta, settings.image_model, cache) if product_shots else None
    return Services(
        sam=SamSegmenter(meta, settings.sam_model, cache),
        spark=spark,
        listing=ListingChain([*partners, ListingProvider("meta", spark)]),
        photographer=photographer,
    )


@dataclass(frozen=True, slots=True)
class _Found:
    still: Still
    media_path: str
    segment: Segment
    calls: tuple[ModelCall, ...]
    phrase: str
    slug: str


def _best(segments: Sequence[Segment]) -> Segment | None:
    """The most confident segment for a phrase (largest when confidence is missing)."""
    if not segments:
        return None
    return max(segments, key=lambda s: (s.confidence or 0.0, s.box.area))


def _best_sightings(found: list[_Found]) -> list[_Found]:
    """A video shows an item in several frames: keep its best view (largest mask) before any
    per-item call, so each item is listed and photographed once."""
    best: dict[tuple[str, str], _Found] = {}
    for sighting in found:
        if sighting.still.frame_index is None:
            continue
        key = (str(sighting.still.source), sighting.phrase.casefold())
        kept = best.get(key)
        if kept is None or sighting.segment.mask.sum() > kept.segment.mask.sum():
            best[key] = sighting
    keep = {id(s) for s in best.values()}
    return [f for f in found if f.still.frame_index is None or id(f) in keep]


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
        self.warnings: list[str] = []
        self.partial: dict[str, Item] = {}  # items listed so far, for live display
        self._next_number = 0

    def _too_small(self, segment: Segment) -> bool:
        box = segment.box
        short_side = min(box.right - box.left, box.bottom - box.top)
        return (
            short_side < self.settings.min_item_side_px
            or int(segment.mask.sum()) < self.settings.min_item_pixels
        )

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        self.progress(f"warning: {message}")

    async def run(
        self, inputs: Sequence[Path], out_dir: Path, language: Language | None = None
    ) -> ListingPack:
        language = language or self.settings.default_language
        files = collect_inputs(inputs)
        if not files:
            raise ValueError("no supported photos or videos in the inputs")
        self.warnings = []
        (out_dir / "media").mkdir(parents=True, exist_ok=True)

        self.partial = {}
        self._next_number = 0
        per_file = await asyncio.gather(*(self._file(f, out_dir, language) for f in files))
        media_paths = [path for media, _ in per_file for path in media]
        if not media_paths:
            raise ValueError("every photo or frame was too small: " + "; ".join(self.warnings))
        items = sorted((item for _, found in per_file for item in found), key=lambda i: i.item_id)
        pack = ListingPack(
            run_id=f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}",
            created_at=datetime.now(UTC),
            media=media_paths,
            items=_drop_video_repeats(items),
            warnings=self.warnings,
        )
        save_pack(out_dir, pack)
        self.progress(f"done: {len(pack.items)} items")
        return pack

    async def _file(
        self, file: Path, out_dir: Path, language: Language
    ) -> tuple[list[str], list[Item]]:
        """One photo or video: scan its stills in order, then list its items in parallel.

        Files run side by side, and a file's listings start as soon as its own scan ends.
        """
        self.progress(f"reading {file.name}")
        media: list[str] = []
        found: list[_Found] = []
        seen: list[str] = []  # phrases from earlier frames of this video
        for still in await asyncio.to_thread(stills, file, self.settings.max_image_side_px):
            min_side = self.settings.min_still_side_px
            if min(still.image.size) < min_side:
                width, height = still.image.size
                self._warn(
                    f"{file.name}: {width}x{height} px is too small (under {min_side} px"
                    " on a side); use a higher-resolution photo or video"
                )
                continue
            media_path = f"media/{still.stem}.jpg"
            (out_dir / media_path).write_bytes(encode(still.image, "JPEG"))
            media.append(media_path)
            hits = await self._detect(still, media_path, seen)
            found.extend(hits)
            seen.extend(h.phrase for h in hits if h.phrase not in seen)
        detections = _merge_duplicates(_best_sightings(found))
        self.items_total += len(detections)
        if detections:
            self.progress(f"{file.name}: writing listings for {len(detections)} items")
        items = await asyncio.gather(
            *(self._item(self._new_item_id(), f, out_dir, language) for f in detections)
        )
        return media, list(items)

    def _new_item_id(self) -> str:
        self._next_number += 1
        return f"item-{self._next_number:03d}"

    async def _detect(self, still: Still, media_path: str, seen: Sequence[str]) -> list[_Found]:
        inventory = await self.services.spark.inventory(still.image, seen)
        unique = {item.phrase: item for item in inventory.value.items}
        self.progress(f"{media_path}: looking for {', '.join(unique) or 'nothing'}")

        async def segment(phrase: str) -> tuple[Segment | None, ModelCall]:
            async with self._limit:
                run = await self.services.sam.segment(still.image, phrase)
            return _best(run.segments), run.call

        results = await asyncio.gather(*(segment(p) for p in unique))
        hits = []
        for item, (seg, call) in zip(unique.values(), results, strict=True):
            if seg is None:
                continue
            if self._too_small(seg):
                pixels = int(seg.mask.sum())
                self._warn(
                    f"{media_path}: '{item.phrase}' covers only {pixels} px "
                    f"({seg.box.right - seg.box.left}x{seg.box.bottom - seg.box.top} box), "
                    "too few pixels to judge it; take a closer photo"
                )
                continue
            hits.append((seg, call, item.phrase, slugify(item.short_name or item.phrase)))
        share = _split_cost(inventory.call, len(hits))
        return [
            _Found(still, media_path, seg, (share, call), phrase, slug)
            for seg, call, phrase, slug in hits
        ]

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
            analysis, failed = await self.services.listing.analyze(item_crop, language)
        if failed:
            self.progress(f"{stem}: listing fell back past {', '.join(failed)}")
        canvas = studio_canvas(image, found.segment.mask, box)
        studio = _save(out_dir, f"shots/{stem}.jpg", encode(canvas.real, "JPEG", quality=92))
        item = Item(
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
            product_shots=[studio],
            calls=[*found.calls, analysis.call],
        )
        self.partial[item_id] = item  # the card can show now; a completed shot may follow
        completed, shot_calls = await self._completed_shot(stem, canvas, analysis.value, out_dir)
        item = item.model_copy(
            update={
                "product_shots": [*completed, studio],
                "calls": [*item.calls, *shot_calls],
            }
        )
        self.partial[item_id] = item
        self.items_done += 1
        return item

    async def _completed_shot(
        self, stem: str, canvas: StudioCanvas, analysis: ItemAnalysis, out_dir: Path
    ) -> tuple[list[str], list[ModelCall]]:
        """The item with its hidden parts AI-completed, when that can be trusted.

        The studio shot of the real pixels always exists. When part of the item is hidden,
        Muse Image fills only the hidden areas, every visible pixel is restored from the photo,
        and the result is kept only if the model did not move or redraw the item (drift) and
        Muse Spark sees no extra object or invented damage. One corrected retry.
        """
        photographer = self.services.photographer
        if photographer is None or canvas.hidden_share <= HIDDEN_MIN_SHARE:
            return [], []
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
                    return [_save(out_dir, f"shots/{stem}-completed.jpg", jpeg)], calls
                rejected = check.value.reason
        except (
            OpenAIError,
            EmptyImageResponseError,
            OfflineCacheMissError,
            SparkReplyError,
        ) as error:
            rejected = type(error).__name__
        self.progress(f"{stem}: real-pixel shot only (completion rejected: {rejected})")
        return [], calls


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
