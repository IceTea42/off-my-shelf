"""Fake Meta API client for offline tests: answers by endpoint and requested schema."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from off_my_shelf.mocks import load_mock_pack
from off_my_shelf.spark import SceneInventory

# Two objects in a 320x334 image, in SAM 3.1's token format (from meta-sam-parser's docs).
SAM_SAMPLE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)
SAM_SIZE = (320, 334)
ANALYSIS_JSON = load_mock_pack().items[0].analysis.model_dump_json()


class FakeMetaClient:
    """Stands in for ``AsyncOpenAI``: SAM streams, Spark structured replies."""

    def __init__(self, phrases: list[str], sam_text: str = SAM_SAMPLE) -> None:
        inventory = SceneInventory.model_validate(
            {"items": [{"phrase": p, "label": p.title()} for p in phrases]}
        )
        self.inventory_json = inventory.model_dump_json()
        self.sam_text = sam_text
        self.chat_calls: list[dict[str, Any]] = []
        self.sam_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)

    async def _chat(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        self.chat_calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        content = self.inventory_json if name == "SceneInventory" else ANALYSIS_JSON
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=300),
        )

    async def _responses(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:  # noqa: ANN401
        self.sam_calls.append(kwargs)
        text = self.sam_text

        async def stream() -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(type="response.output_text.delta", delta=text)
            yield SimpleNamespace(type="response.completed")

        return stream()
