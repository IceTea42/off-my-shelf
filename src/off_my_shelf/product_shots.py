"""Muse Image: complete the hidden parts of an item, and nothing else.

The visible item always comes from the photo's real pixels (``imaging.studio_canvas``). Muse
Image only fills the magenta areas, where the item was covered by something else; afterwards
``imaging.pixel_lock`` restores every visible pixel, so only hidden parts can be AI-drawn.
"""

import base64
import io
import time

from openai import AsyncOpenAI
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import IMAGE_GENERATION_USD
from off_my_shelf.imaging import encode
from off_my_shelf.schema import ModelCall, PipelineStep

FILL_PROMPT = (
    "This is a product photo of one item: {name}, on a white background. The magenta areas "
    "are parts of the item that were hidden behind other objects. Fill only the magenta areas "
    "as the natural, undamaged continuation of this item, matching its shape, material, colour "
    "and lighting. Keep every other pixel exactly as it is: same framing, size and position. "
    "Real defects of the item: {defects}; do not add any other damage, dust, dirt, text or "
    "object."
)
STRICT_HINT = "A previous attempt was rejected: {reason}. Fix exactly that. "
SHOT_SIZE = "1024x1024"


class EmptyImageResponseError(RuntimeError):
    """The image endpoint answered without image data."""


class ProductPhotographer:
    """Completes hidden parts on a studio canvas; ``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def complete(
        self,
        to_fill: Image.Image,
        name: str,
        defects: list[str],
        rejected_because: str | None = None,
    ) -> tuple[Image.Image, ModelCall]:
        """The model's completed canvas (not yet pixel-locked) and the call record."""
        prompt = FILL_PROMPT.format(name=name, defects="; ".join(defects) if defects else "none")
        if rejected_because:
            prompt = STRICT_HINT.format(reason=rejected_because) + prompt
        source = encode(to_fill, "PNG")
        key = cache_key("image", self.model, prompt, SHOT_SIZE, source)
        data = self.cache.get_bytes(key)
        latency_ms: int | None = None
        if data is None:
            if self.client is None:
                raise OfflineCacheMissError(PipelineStep.PRODUCT_SHOT.value, key)
            start = time.perf_counter()
            response = await self.client.images.edit(
                model=self.model,
                image=("item.png", source, "image/png"),
                prompt=prompt,
                size=SHOT_SIZE,
                n=1,
                response_format="b64_json",
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            encoded = response.data[0].b64_json if response.data else None
            if not encoded:
                raise EmptyImageResponseError(f"{self.model} returned no image")
            data = base64.b64decode(encoded)
            self.cache.put_bytes(key, data)
        with Image.open(io.BytesIO(data)) as raw:
            completed = raw.convert("RGB")
        call = ModelCall(
            step=PipelineStep.PRODUCT_SHOT,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            cost_usd=IMAGE_GENERATION_USD,
            latency_ms=latency_ms,
        )
        return completed, call
