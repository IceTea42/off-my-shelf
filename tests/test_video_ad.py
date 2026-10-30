"""Video ads: real-shot frames, overlays, offline and live hooks, web and CLI wiring."""

import shutil
from pathlib import Path

import av
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from fakes import FakeMetaClient, fake_services
from off_my_shelf import video_ad, web
from off_my_shelf.cache import DiskCache
from off_my_shelf.cli import main
from off_my_shelf.config import Settings
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack, load_pack
from off_my_shelf.spark import SparkAnalyzer


@pytest.fixture(autouse=True)
def short_ads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_ad, "FPS", 4)
    monkeypatch.setattr(video_ad, "SECONDS", 1)


def test_wrap_shortens_long_titles() -> None:
    font = video_ad._font(56)
    lines = video_ad._wrap("word " * 60, font, 900, 2)
    assert len(lines) == 2 and lines[-1].endswith("…")
    assert all(font.getlength(line) <= 900 for line in lines)


async def test_item_ad_offline_and_live(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    item = load_mock_pack().items[0]

    rel, call = await video_ad.make_item_ad(None, pack_dir, item)
    assert rel == "ads/item-001.mp4" and call is None
    with av.open(str(pack_dir / rel)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
    assert (stream.width, stream.height) == video_ad.AD_SIZE and len(frames) == 4

    fake = FakeMetaClient([])
    spark = SparkAnalyzer(fake, "muse-spark-1.3", DiskCache(tmp_path / "cache"), 4000)  # type: ignore[arg-type]
    _, live = await video_ad.make_item_ad(spark, pack_dir, item)
    assert live is not None and live.cost_usd is not None
    assert "Lampe de bureau" in str(fake.chat_calls[0]["messages"])


def test_web_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMetaClient([])
    cache = DiskCache(tmp_path / "cache")
    monkeypatch.setattr(
        web,
        "build_services",
        lambda *_, **__: fake_services(fake, cache),
    )
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        meta_api_key=SecretStr("test-key"),
    )
    with TestClient(web.create_app(settings)) as client:
        run_id = client.post("/api/demo").json()["run_id"]
        first = client.post(f"/api/runs/{run_id}/items/item-002/ad").json()
        assert first == {"video": "ads/item-002.mp4"}
        assert client.post(f"/api/runs/{run_id}/items/item-002/ad").json() == first
        assert len(fake.chat_calls) == 1
        assert client.get(f"/runs/{run_id}/pack/ads/item-002.mp4").status_code == 200

    pack_dir = tmp_path / "cli-pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    main(["ads", str(pack_dir), "--offline"])
    made = {i.item_id: i.ad_video for i in load_pack(pack_dir).items}
    assert made["item-001"] == "ads/item-001.mp4" and made["item-005"] is None  # recycle: no ad
