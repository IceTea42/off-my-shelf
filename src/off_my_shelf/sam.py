"""SAM 3.1 through the Meta Model API: one noun phrase in, boxes and masks out.

The API streams special-token text (see ``meta_sam_parser``). We cache that raw text, so a
cached run is parsed offline exactly like a live one.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)
from openai import AsyncOpenAI
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import SAM_IMAGE_USD
from off_my_shelf.imaging import BoolMask, PixelBox, data_url, encode
from off_my_shelf.schema import ModelCall, PipelineStep


class SamStreamError(RuntimeError):
    """The segmentation stream ended with an error event."""


@dataclass(frozen=True, slots=True)
class Segment:
    object_id: str
    box: PixelBox
    mask: BoolMask
    """Boolean mask the size of the segmented image."""
    confidence: float | None


@dataclass(frozen=True, slots=True)
class SegmentationRun:
    segments: list[Segment]
    call: ModelCall


def _place_mask(record: SegmentationMaskRecord, width: int, height: int) -> BoolMask:
    """Decode a box-local mask raster and paste it into a full-size boolean mask."""
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    local: BoolMask = raster.reshape(record.mask.height, record.mask.width).astype(bool)
    full = np.zeros((height, width), dtype=bool)
    top, left = int(record.bounds.top), int(record.bounds.left)
    bottom = min(height, top + local.shape[0])
    right = min(width, left + local.shape[1])
    full[top:bottom, left:right] = local[: bottom - top, : right - left]
    return full


def parse_segments(output_text: str, width: int, height: int) -> list[Segment]:
    """Turn SAM's token text for one image into segments, pairing each box with its mask."""
    parser = image_segmentation_format().create_parser()
    parser.push(output_text)
    result = parser.finish(CompletedOutcome()).result
    boxes = {r.object_id: r for r in result.records if isinstance(r, SegmentationBoxRecord)}
    segments = []
    for record in result.records:
        if not isinstance(record, SegmentationMaskRecord) or record.object_id not in boxes:
            continue
        box = boxes[record.object_id]
        pixel_box = PixelBox(
            max(0, int(box.left)),
            max(0, int(box.top)),
            min(width, int(box.right)),
            min(height, int(box.bottom)),
        )
        confidence = box.confidence if box.confidence is not None else record.confidence
        mask = _place_mask(record, width, height)
        segments.append(Segment(record.object_id, pixel_box, mask, confidence))
    return segments


async def _stream_output_text(client: AsyncOpenAI, model: str, phrase: str, image: str) -> str:
    """Collect the output text of one streamed SAM call."""
    message: Any = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_image", "image_url": image},
        ],
    }
    stream: Any = await client.responses.create(
        model=model,
        input=[message],
        stream=True,
        metadata={"include_confidence": "true", "mask_encoding": "lossless"},
    )
    deltas: list[str] = []
    final: str | None = None
    async for event in stream:
        kind = getattr(event, "type", "")
        if kind == "response.output_text.delta":
            deltas.append(event.delta)
        elif kind == "response.output_text.done":
            final = event.text
        elif kind == "response.content_part.done" and getattr(event.part, "type", "") == (
            "output_text"
        ):
            final = event.part.text
        elif kind in {"response.failed", "error"}:
            raise SamStreamError(str(event))
    return final if final is not None else "".join(deltas)


class SamSegmenter:
    """Segments one concept per call; ``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def segment(self, image: Image.Image, phrase: str) -> SegmentationRun:
        payload = encode(image, "JPEG")
        key = cache_key("sam", self.model, phrase, payload)
        cached = self.cache.get_json(key)
        latency_ms: int | None = None
        if isinstance(cached, dict):
            text = str(cached["output_text"])
        elif self.client is None:
            raise OfflineCacheMissError(PipelineStep.SEGMENT.value, key)
        else:
            start = time.perf_counter()
            text = await _stream_output_text(
                self.client, self.model, phrase, data_url(payload, "JPEG")
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            self.cache.put_json(key, {"phrase": phrase, "output_text": text})
        call = ModelCall(
            step=PipelineStep.SEGMENT,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            cost_usd=SAM_IMAGE_USD,
            latency_ms=latency_ms,
        )
        return SegmentationRun(parse_segments(text, image.width, image.height), call)
