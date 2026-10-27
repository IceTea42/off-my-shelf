"""S2 listing pack: product shots, voice notes and the HTML/zip export."""

import io
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fakes import FakeMetaClient
from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.cli import main
from off_my_shelf.export import export_pack, listing_text, render_html
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack
from off_my_shelf.product_shots import ProductPhotographer
from off_my_shelf.voice import AudioFormatError, Transcriber, to_api_wav


def _wav(rate: int, channels: int, width: int, seconds: float = 0.5) -> bytes:
    count = int(rate * seconds)
    tone = np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, count))
    if width == 1:
        samples = ((tone * 100) + 128).astype(np.uint8).tobytes()
    elif width == 3:
        ints = (tone * 2**22).astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        samples = ints.tobytes()
    else:
        samples = (tone * 2 ** (8 * width - 2)).astype(f"<i{width}").tobytes()
    frames = b"".join(samples[i : i + width] * channels for i in range(0, len(samples), width))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("rate", "channels", "width"), [(44_100, 2, 2), (8_000, 1, 1), (48_000, 1, 3), (16_000, 1, 4)]
)
def test_to_api_wav_normalises(rate: int, channels: int, width: int) -> None:
    out = to_api_wav(_wav(rate, channels, width))
    with wave.open(io.BytesIO(out)) as result:
        assert (result.getnchannels(), result.getsampwidth(), result.getframerate()) == (
            1,
            2,
            16_000,
        )
        assert result.getnframes() == pytest.approx(8_000, abs=2)
        peak = np.abs(np.frombuffer(result.readframes(result.getnframes()), "<i2")).max()
    assert peak > 1000


def test_to_api_wav_rejects_other_audio() -> None:
    with pytest.raises(AudioFormatError):
        to_api_wav(b"OggS not a wav")


async def test_transcriber_live_then_cached(tmp_path: Path) -> None:
    client = FakeMetaClient([])
    cache = DiskCache(tmp_path)
    text, call = await Transcriber(client, "muse-voice-transcribe-1.0", cache).transcribe(  # type: ignore[arg-type]
        _wav(44_100, 2, 2)
    )
    assert text == client.transcript
    assert call.cost_usd == pytest.approx(0.18 * 1.5 / 3600)
    path, request = client.post_calls[0]
    assert path == "/asr/transcribe"
    assert [name for name, _ in request["files"]] == ["request", "audio"]

    again, cached = await Transcriber(None, "muse-voice-transcribe-1.0", cache).transcribe(
        _wav(44_100, 2, 2)
    )
    assert again == text and cached.cached


async def test_completion_prompt_and_cache(tmp_path: Path) -> None:
    client = FakeMetaClient([])
    cache = DiskCache(tmp_path)
    canvas = Image.new("RGB", (64, 64), "white")
    photographer = ProductPhotographer(client, "muse-image-1.0", cache)  # type: ignore[arg-type]
    filled, call = await photographer.complete(canvas, "white dinner plate", ["chipped rim"])
    assert filled.size == (64, 64) and call.cost_usd == pytest.approx(0.01)
    prompt = client.image_calls[0]["prompt"]
    assert "white dinner plate" in prompt and "chipped rim" in prompt
    assert "Fill only the magenta areas" in prompt and "Keep every other pixel" in prompt

    again, cached = await ProductPhotographer(None, "muse-image-1.0", cache).complete(
        canvas, "white dinner plate", ["chipped rim"]
    )
    assert cached.cached and again.size == filled.size
    await photographer.complete(canvas, "white dinner plate", [], rejected_because="a dent")
    assert client.image_calls[1]["prompt"].startswith("A previous attempt was rejected: a dent")
    with pytest.raises(OfflineCacheMissError):
        await ProductPhotographer(None, "muse-image-1.0", cache).complete(canvas, "other", [])


def test_render_html_escapes_and_summarises() -> None:
    pack = load_mock_pack()
    first = pack.items[0].model_copy(deep=True)
    first.analysis.listing.title = "Lamp <script>alert(1)</script>"
    html = render_html(pack.model_copy(update={"items": [first, *pack.items[1:]]}))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert f"{len(pack.items)} items" in html
    assert "Clean shot, real pixels" in html or "AI-completed" in html
    assert "data:image/svg+xml,%3Csvg" in html


def test_listing_text_has_price_and_tags() -> None:
    texts = listing_text(load_mock_pack())
    assert set(texts) == {item.item_id for item in load_mock_pack().items}
    assert all("Price:" in t and "Tags:" in t for t in texts.values())


def test_export_pack_writes_page_text_and_zip(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    archive = export_pack(pack_dir)
    assert archive == tmp_path / "pack.zip"
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert {"index.html", "listing_pack.json", "listings/item-001.txt"} <= names
    assert "crops/item-001.png" in names


def test_cli_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    main(["export", str(pack_dir)])
    assert "pack.zip" in capsys.readouterr().out
