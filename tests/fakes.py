"""Fake Meta API client for offline tests: answers by endpoint and requested schema."""

import base64
import io
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from PIL import Image

from off_my_shelf.cache import DiskCache
from off_my_shelf.mocks import load_mock_pack
from off_my_shelf.pipeline import Services
from off_my_shelf.product_shots import ProductPhotographer
from off_my_shelf.sam import SamSegmenter
from off_my_shelf.spark import ListingChain, ListingProvider, SceneInventory, SparkAnalyzer

# Two objects in a 320x334 image, in SAM 3.1's token format (from meta-sam-parser's docs).
SAM_SAMPLE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)
SAM_SIZE = (320, 334)
ANALYSIS_JSON = load_mock_pack().items[0].analysis.model_dump_json()
TRANSLATION_JSON = (
    '{"title": "Black articulated desk lamp, good condition", '
    '"description": "Desk lamp with an articulated arm. Light scratches on the base.", '
    '"tags": ["lamp", "desk"]}'
)
SHOT_OK = '{"same_item": true, "extra_objects": false, "added_defects": false, "reason": "ok"}'
SHOT_BAD = '{"same_item": true, "extra_objects": false, "added_defects": true, "reason": "dent"}'


def webp_bytes(size: tuple[int, int] = (32, 32), colour: str = "darkred") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "WEBP")
    return buffer.getvalue()


class FakeMetaClient:
    """Stands in for ``AsyncOpenAI``: SAM streams, Spark structured replies."""

    def __init__(self, phrases: list[str], sam_text: str = SAM_SAMPLE) -> None:
        inventory = SceneInventory.model_validate(
            {"items": [{"phrase": p, "label": p.title(), "short_name": p} for p in phrases]}
        )
        self.inventory_json = inventory.model_dump_json()
        self.sam_text = sam_text
        self.sam_misses: set[str] = set()  # phrases SAM finds nothing for
        self.chat_calls: list[dict[str, Any]] = []
        self.sam_calls: list[dict[str, Any]] = []
        self.image_calls: list[dict[str, Any]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.transcript = "It works but the cable is a bit worn."
        self.post_error: Exception | None = None
        self.shot_checks = [SHOT_OK]
        self.redraw = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)
        self.images = SimpleNamespace(edit=self._image_edit)

    async def _image_edit(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        self.image_calls.append(kwargs)
        if self.redraw:  # a model that ignores the canvas and draws something else
            picture = webp_bytes((32 + len(self.image_calls), 32))
        else:  # a faithful fill: the canvas back, one pixel different per call
            with Image.open(io.BytesIO(kwargs["image"][1])) as canvas:
                filled = canvas.convert("RGB")
            filled.putpixel((0, 0), (len(self.image_calls), 0, 0))
            buffer = io.BytesIO()
            filled.save(buffer, "PNG")
            picture = buffer.getvalue()
        data = base64.b64encode(picture).decode("ascii")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=data)])

    async def post(self, path: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        self.post_calls.append((path, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return {"sessionId": "s1", "transcript": self.transcript, "audioDurationMs": 1500}

    async def _chat(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        self.chat_calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        if name == "SceneInventory":
            content = self.inventory_json
        elif name == "AdCopy":
            content = '{"hook": "Light up your desk for less"}'
        elif name == "ListingText":
            content = TRANSLATION_JSON
        elif name == "ShotCheck":
            content = self.shot_checks[min(len(self.image_calls), len(self.shot_checks)) - 1]
        else:
            content = ANALYSIS_JSON
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=300),
        )

    async def _responses(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:  # noqa: ANN401
        self.sam_calls.append(kwargs)
        phrase = kwargs["input"][0]["content"][0]["text"]
        text = "" if phrase in self.sam_misses else self.sam_text

        async def stream() -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(type="response.output_text.delta", delta=text)
            yield SimpleNamespace(type="response.completed")

        return stream()


def fake_services(client: FakeMetaClient, cache: DiskCache, photographer: bool = False) -> Services:
    """Services wired to the fake client, with the default settings' models and reasoning
    effort (so their cache matches ``build_services``); listings use the Meta-only chain."""
    spark = SparkAnalyzer(client, "muse-spark-1.3", cache, 4000, "low")  # type: ignore[arg-type]
    return Services(
        sam=SamSegmenter(client, "sam-3.1", cache),  # type: ignore[arg-type]
        spark=spark,
        listing=ListingChain([ListingProvider("meta", spark)]),
        photographer=(
            ProductPhotographer(client, "muse-image-1.0", cache)  # type: ignore[arg-type]
            if photographer
            else None
        ),
    )
