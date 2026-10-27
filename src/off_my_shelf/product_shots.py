"""Muse Image: a clean product shot from an item's cut-out, keeping its real look and wear."""

import base64
import time

from openai import AsyncOpenAI
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import IMAGE_GENERATION_USD
from off_my_shelf.imaging import encode
from off_my_shelf.schema import ModelCall, PipelineStep

SHOT_PROMPT = (
    "Turn this cut-out into a clean e-commerce product photo of the exact same item: "
    "centred on a plain light grey studio background with a soft natural shadow. Keep its real "
    "shape, colours, materials, wear and defects unchanged. Do not add text, logos, props or "
    "other objects."
)
SHOT_SIZE = "1024x1024"


class EmptyImageResponseError(RuntimeError):
    """The image endpoint answered without image data."""


class ProductPhotographer:
    """One product shot per item; ``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def shoot(self, item_cutout: Image.Image) -> tuple[bytes, ModelCall]:
        """WebP bytes of the product shot, and the call record."""
        source = encode(item_cutout, "PNG")
        key = cache_key("image", self.model, SHOT_PROMPT, SHOT_SIZE, source)
        shot = self.cache.get_bytes(key)
        latency_ms: int | None = None
        if shot is None:
            if self.client is None:
                raise OfflineCacheMissError(PipelineStep.PRODUCT_SHOT.value, key)
            start = time.perf_counter()
            response = await self.client.images.edit(
                model=self.model,
                image=("item.png", source, "image/png"),
                prompt=SHOT_PROMPT,
                size=SHOT_SIZE,
                n=1,
                response_format="b64_json",
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            data = response.data[0].b64_json if response.data else None
            if not data:
                raise EmptyImageResponseError(f"{self.model} returned no image")
            shot = base64.b64decode(data)
            self.cache.put_bytes(key, shot)
        call = ModelCall(
            step=PipelineStep.PRODUCT_SHOT,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            cost_usd=IMAGE_GENERATION_USD,
            latency_ms=latency_ms,
        )
        return shot, call
