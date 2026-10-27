"""S1 core pipeline: photos or videos in, a validated ``ListingPack`` on disk out.

For each still: Muse Spark lists the items (noun phrases) -> SAM 3.1 segments each phrase ->
overlapping detections are merged -> each item crop gets a full Muse Spark listing.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings
from off_my_shelf.imaging import PixelBox, crop, encode, mask_image
from off_my_shelf.media import Still, collect_inputs, stills
from off_my_shelf.meta_client import make_client
from off_my_shelf.mocks import PACK_FILENAME
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
from off_my_shelf.spark import SparkAnalyzer

DUPLICATE_IOU = 0.6
CROP_PADDING = 0.08
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Services:
    sam: SamSegmenter
    spark: SparkAnalyzer


def build_services(settings: Settings, offline: bool = False) -> Services:
    """Live services, or cache-only ones when ``offline`` (no key, no network, no cost)."""
    client = None if offline else make_client(settings)
    cache = DiskCache(settings.cache_dir)
    return Services(
        sam=SamSegmenter(client, settings.sam_model, cache),
        spark=SparkAnalyzer(client, settings.spark_model, cache, settings.max_output_tokens),
    )


@dataclass(frozen=True, slots=True)
class _Found:
    still: Still
    media_path: str
    segment: Segment
    calls: tuple[ModelCall, ...]


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
        (out_dir / PACK_FILENAME).write_text(
            pack.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        self.progress(f"done: {len(pack.items)} items")
        return pack

    async def _detect(self, still: Still, media_path: str) -> list[_Found]:
        inventory = await self.services.spark.inventory(still.image)
        phrases = list(dict.fromkeys(item.phrase for item in inventory.value.items))
        self.progress(f"{media_path}: looking for {', '.join(phrases) or 'nothing'}")

        async def segment(phrase: str) -> tuple[Segment | None, ModelCall]:
            async with self._limit:
                run = await self.services.sam.segment(still.image, phrase)
            return _best(run.segments), run.call

        results = await asyncio.gather(*(segment(p) for p in phrases))
        hits = [(seg, call) for seg, call in results if seg is not None]
        share = _split_cost(inventory.call, len(hits))
        return [_Found(still, media_path, seg, (share, call)) for seg, call in hits]

    async def _item(self, item_id: str, found: _Found, out_dir: Path, language: Language) -> Item:
        image = found.still.image
        box: PixelBox = found.segment.box.padded(CROP_PADDING, image.width, image.height)
        item_crop = crop(image, box)
        crop_path, mask_path = f"crops/{item_id}.jpg", f"masks/{item_id}.png"
        for rel, data in (
            (crop_path, encode(item_crop, "JPEG")),
            (mask_path, encode(mask_image(found.segment.mask), "PNG")),
        ):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(data)

        async with self._limit:
            analysis = await self.services.spark.analyze(item_crop, language)
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
            calls=[*found.calls, analysis.call],
        )


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
