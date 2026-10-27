"""SAM 3.1 client: token parsing, mask placement, cache and stream handling."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.imaging import PixelBox
from off_my_shelf.sam import SamSegmenter, SamStreamError, parse_segments

# Two objects in a 320x334 image, in the API's token format (from meta-sam-parser's docs).
SAMPLE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)
WIDTH, HEIGHT = 320, 334


def test_parse_segments_pairs_boxes_and_masks() -> None:
    segments = parse_segments(SAMPLE, WIDTH, HEIGHT)
    assert [s.object_id for s in segments] == ["0", "1"]
    first = segments[0]
    assert first.box == PixelBox(211, 228, 271, 255)
    assert first.mask.shape == (HEIGHT, WIDTH)
    assert int(first.mask.sum()) == 1539
    ys, xs = first.mask.nonzero()
    assert xs.min() >= 211 and xs.max() < 271
    assert ys.min() >= 228 and ys.max() < 255
    assert first.confidence is None


def test_parse_segments_empty_output() -> None:
    assert parse_segments("", WIDTH, HEIGHT) == []


class _FakeResponses:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> AsyncIterator[object]:
        self.calls.append(kwargs)

        async def stream() -> AsyncIterator[object]:
            for event in self.events:
                yield event

        return stream()


def _client(events: list[Any]) -> Any:  # noqa: ANN401 - stands in for AsyncOpenAI
    return SimpleNamespace(responses=_FakeResponses(events))


def _delta_events(text: str) -> list[Any]:
    half = len(text) // 2
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text[:half]),
        SimpleNamespace(type="response.output_text.delta", delta=text[half:]),
        SimpleNamespace(type="response.completed"),
    ]


async def test_live_call_then_cache_hit(tmp_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    cache = DiskCache(tmp_path)
    client = _client(_delta_events(SAMPLE))
    live = await SamSegmenter(client, "sam-3.1", cache).segment(image, "box")
    assert len(live.segments) == 2
    assert live.call.cached is False and live.call.latency_ms is not None
    sent = client.responses.calls[0]
    assert sent["model"] == "sam-3.1" and sent["stream"] is True
    assert sent["input"][0]["content"][0] == {"type": "input_text", "text": "box"}

    offline = await SamSegmenter(None, "sam-3.1", cache).segment(image, "box")
    assert offline.call.cached is True
    assert offline.call.cache_key == live.call.cache_key
    assert [s.box for s in offline.segments] == [s.box for s in live.segments]


async def test_content_part_done_text_wins(tmp_path: Path) -> None:
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="garbage"),
        SimpleNamespace(
            type="response.content_part.done",
            part=SimpleNamespace(type="output_text", text=SAMPLE),
        ),
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT))
    run = await SamSegmenter(_client(events), "sam-3.1", DiskCache(tmp_path)).segment(image, "x")
    assert len(run.segments) == 2


async def test_offline_cache_miss(tmp_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    with pytest.raises(OfflineCacheMissError):
        await SamSegmenter(None, "sam-3.1", DiskCache(tmp_path)).segment(image, "mug")


async def test_stream_error_raises(tmp_path: Path) -> None:
    client = _client([SimpleNamespace(type="error", message="boom")])
    image = Image.new("RGB", (WIDTH, HEIGHT))
    with pytest.raises(SamStreamError):
        await SamSegmenter(client, "sam-3.1", DiskCache(tmp_path)).segment(image, "mug")
