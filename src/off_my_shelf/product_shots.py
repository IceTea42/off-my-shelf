"""Muse Image: a clean product shot from an item's cut-out, keeping its real look and wear.

The cut-out is flattened onto a plain background before sending (transparent pixels where the
item was hidden came back as black or empty holes), the model is asked to complete hidden parts
plausibly, and the result is flattened onto white and saved as JPEG, which every marketplace
accepts.
"""

import base64
import io
import time

from openai import AsyncOpenAI
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import IMAGE_GENERATION_USD
from off_my_shelf.imaging import encode, flatten
from off_my_shelf.schema import ModelCall, PipelineStep

SHOT_PROMPT = (
    "Turn this photo into a clean e-commerce product photo of exactly one item: {name}. "
    "Show it alone, whole and centred on a plain light grey studio background with a soft "
    "natural shadow, in the same colours and materials as in the photo. "
    "{hidden}"
    "Real defects to keep visible: {defects}. Never add any other damage, dents, scratches, "
    "stains, dust or dirt. Do not add any other object, prop, text or logo."
)
HIDDEN_HINT = (
    "The magenta areas are parts hidden behind other objects in the original photo: fill them "
    "as the natural, undamaged continuation of the item, never as damage and never magenta. "
)
STRICT_HINT = "A previous attempt was rejected: {reason}. Fix exactly that. "
SHOT_SIZE = "1024x1024"


class EmptyImageResponseError(RuntimeError):
    """The image endpoint answered without image data."""


class ProductPhotographer:
    """One product shot per item; ``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def shoot(
        self,
        source_image: Image.Image,
        name: str,
        defects: list[str],
        has_hidden: bool = False,
        rejected_because: str | None = None,
    ) -> tuple[bytes, ModelCall]:
        """JPEG bytes of the product shot, and the call record.

        ``source_image`` comes from ``imaging.shot_input``: the item on a plain background,
        hidden parts in magenta. ``rejected_because`` asks for a corrected second attempt.
        """
        prompt = SHOT_PROMPT.format(
            name=name,
            hidden=HIDDEN_HINT if has_hidden else "",
            defects="; ".join(defects) if defects else "none",
        )
        if rejected_because:
            prompt = STRICT_HINT.format(reason=rejected_because) + prompt
        source = encode(source_image, "PNG")
        key = cache_key("image", self.model, prompt, SHOT_SIZE, source)
        shot = self.cache.get_bytes(key)
        latency_ms: int | None = None
        if shot is None:
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
            data = response.data[0].b64_json if response.data else None
            if not data:
                raise EmptyImageResponseError(f"{self.model} returned no image")
            shot = base64.b64decode(data)
            self.cache.put_bytes(key, shot)
        with Image.open(io.BytesIO(shot)) as raw:
            jpeg = encode(flatten(raw, (255, 255, 255)), "JPEG", quality=92)
        call = ModelCall(
            step=PipelineStep.PRODUCT_SHOT,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            cost_usd=IMAGE_GENERATION_USD,
            latency_ms=latency_ms,
        )
        return jpeg, call
