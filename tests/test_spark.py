"""Muse Spark structured calls: schema shaping, validation, repair, cache and language."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.mocks import load_mock_pack
from off_my_shelf.schema import ItemAnalysis, Language
from off_my_shelf.spark import (
    SceneInventory,
    SparkAnalyzer,
    SparkReplyError,
    strict_json_schema,
)

VALID = load_mock_pack().items[0].analysis.model_dump_json()


class _FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        self.calls.append(kwargs)
        content = self.replies[len(self.calls) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )


def _analyzer(tmp_path: Path, replies: list[str] | None) -> tuple[SparkAnalyzer, Any]:
    client = None
    if replies is not None:
        client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(replies)))
    analyzer = SparkAnalyzer(client, "muse-spark-1.3", DiskCache(tmp_path), max_tokens=4000)  # type: ignore[arg-type]
    return analyzer, client


def _image() -> Image.Image:
    return Image.new("RGB", (64, 48), "gray")


def test_strict_schema_is_flat_and_fully_required() -> None:
    schema = strict_json_schema(ItemAnalysis)
    text = str(schema)
    assert "$ref" not in text and "$defs" not in text and "'default'" not in text
    assert schema["required"] == list(schema["properties"])
    identification = schema["properties"]["identification"]
    assert identification["additionalProperties"] is False
    assert "brand" in identification["required"]


async def test_analyze_valid_reply_then_offline_cache(tmp_path: Path) -> None:
    analyzer, client = _analyzer(tmp_path, [VALID])
    result = await analyzer.analyze(_image(), Language.FR, note="works fine")
    assert result.value.identification.name
    assert result.call.input_tokens == 1000 and result.call.cost_usd == pytest.approx(0.00210)
    sent = client.chat.completions.calls[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert "French" in sent["messages"][0]["content"]
    assert "works fine" in sent["messages"][1]["content"][0]["text"]

    offline, _ = _analyzer(tmp_path, None)
    again = await offline.analyze(_image(), Language.FR, note="works fine")
    assert again.call.cached is True
    assert again.value == result.value


async def test_language_is_forced_to_the_request(tmp_path: Path) -> None:
    analyzer, _ = _analyzer(tmp_path, [VALID])
    result = await analyzer.analyze(_image(), Language.TR)
    assert result.value.listing.language is Language.TR


async def test_invalid_reply_is_repaired_once(tmp_path: Path) -> None:
    analyzer, client = _analyzer(tmp_path, ['{"not": "valid"}', VALID])
    result = await analyzer.analyze(_image(), Language.EN)
    assert result.value.listing.title
    assert len(client.chat.completions.calls) == 2
    assert result.call.input_tokens == 2000


async def test_two_invalid_replies_raise(tmp_path: Path) -> None:
    analyzer, _ = _analyzer(tmp_path, ["{}", "{}"])
    with pytest.raises(SparkReplyError):
        await analyzer.analyze(_image(), Language.EN)


async def test_inventory(tmp_path: Path) -> None:
    reply = SceneInventory.model_validate(
        {"items": [{"phrase": "desk lamp", "label": "Black desk lamp"}]}
    ).model_dump_json()
    analyzer, _ = _analyzer(tmp_path, [reply])
    result = await analyzer.inventory(_image())
    assert [item.phrase for item in result.value.items] == ["desk lamp"]


async def test_offline_cache_miss(tmp_path: Path) -> None:
    analyzer, _ = _analyzer(tmp_path, None)
    with pytest.raises(OfflineCacheMissError):
        await analyzer.inventory(_image())
