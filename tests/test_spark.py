"""Muse Spark structured calls: schema shaping, validation, repair, cache and language."""

import json
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
    fit,
    json_skeleton,
    parse_reply,
    strict_json_schema,
    wire_schema,
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
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            ],
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
    # A property literally named "title" must survive annotation stripping.
    assert "title" in schema["properties"]["listing"]["required"]


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


async def test_two_invalid_replies_raise_and_are_saved(tmp_path: Path) -> None:
    analyzer, _ = _analyzer(tmp_path, ["{}", "not json"])
    with pytest.raises(SparkReplyError, match="failures"):
        await analyzer.analyze(_image(), Language.EN)
    saved = list((tmp_path / "failures").rglob("*.json"))
    assert len(saved) == 1
    assert "not json" in saved[0].read_text(encoding="utf-8")


def test_wire_schema_drops_limits_but_keeps_shape() -> None:
    strict = strict_json_schema(ItemAnalysis)
    wire = wire_schema(strict)
    text = json.dumps(wire)
    assert "maxLength" not in text and "pattern" not in text and "minimum" not in text
    assert wire["required"] == strict["required"]
    assert '"enum"' in text
    skeleton = json_skeleton(wire)
    assert '"title": string' in skeleton
    assert '"brand": string | null' in skeleton
    assert '"decision": "sell" | "donate" | "recycle"' in skeleton


def test_fit_trims_to_limits_and_drops_junk() -> None:
    strict = strict_json_schema(ItemAnalysis)
    data = json.loads(VALID)
    data["listing"]["title"] = "A " * 60
    data["listing"]["tags"] = ["lamp", "ÿýýýýýýý", *[f"t{i}" for i in range(20)]]
    data["identification"]["confidence"] = 1.7
    data["identification"]["extra"] = "dropped"
    fixed = fit(data, strict)
    assert len(fixed["listing"]["title"]) <= 80
    assert "ÿýýýýýýý" not in fixed["listing"]["tags"]
    assert len(fixed["listing"]["tags"]) == 10
    assert fixed["identification"]["confidence"] == 1.0
    assert "extra" not in fixed["identification"]
    ItemAnalysis.model_validate(fixed)


def test_parse_reply_accepts_code_fences() -> None:
    strict = strict_json_schema(ItemAnalysis)
    value = parse_reply(f"```json\n{VALID}\n```", ItemAnalysis, strict)
    assert value.listing.title


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
