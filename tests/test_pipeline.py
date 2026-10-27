"""End-to-end S1 pipeline on a fake Meta client: media in, validated pack out."""

from pathlib import Path

import cv2
import httpx2
import numpy as np
import pytest
from openai import APIConnectionError
from PIL import Image

from fakes import SAM_SIZE, FakeMetaClient
from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.cli import main
from off_my_shelf.config import Settings
from off_my_shelf.media import UnsupportedMediaError, collect_inputs, stills
from off_my_shelf.mocks import load_pack
from off_my_shelf.pipeline import Pipeline, Services, build_services
from off_my_shelf.product_shots import ProductPhotographer
from off_my_shelf.sam import SamSegmenter
from off_my_shelf.schema import Language
from off_my_shelf.spark import SparkAnalyzer


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, cache_dir=tmp_path / "cache")  # type: ignore[call-arg]


def _services(client: FakeMetaClient, settings: Settings) -> Services:
    cache = DiskCache(settings.cache_dir)
    return Services(
        sam=SamSegmenter(client, settings.sam_model, cache),  # type: ignore[arg-type]
        spark=SparkAnalyzer(client, settings.spark_model, cache, 4000),  # type: ignore[arg-type]
        photographer=ProductPhotographer(client, settings.image_model, cache),  # type: ignore[arg-type]
    )


def _photo(folder: Path, name: str = "shelf.png") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    Image.new("RGB", SAM_SIZE, (200, 190, 170)).save(path)
    return path


async def test_pipeline_writes_a_valid_pack(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    (photos / "notes.txt").write_text("ignored", encoding="utf-8")
    client = FakeMetaClient(phrases=["lamp", "book"])
    messages: list[str] = []
    out = tmp_path / "pack"

    pack = await Pipeline(_services(client, settings), settings, messages.append).run(
        [photos], out, Language.TR
    )

    # Both phrases resolve to the same top object, so the duplicate is merged.
    assert len(client.sam_calls) == 2
    assert len(pack.items) == 1
    item = pack.items[0]
    assert item.analysis.listing.language is Language.TR
    assert pack.media == ["media/shelf.jpg"]
    assert item.product_shots == ["shots/item-001.webp"]
    for rel in (item.detection.crop_path, item.detection.mask_path, *item.product_shots):
        assert rel is not None and (out / rel).is_file()
    assert (out / pack.media[0]).is_file()
    steps = [c.step.value for c in item.calls]
    assert steps == ["analyze", "segment", "analyze", "product_shot"]
    assert client.image_calls[0]["image"][2] == "image/png"
    assert load_pack(out) == pack
    assert messages[-1] == "done: 1 items"


async def test_rerun_is_free_from_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    await Pipeline(_services(FakeMetaClient(["lamp"]), settings), settings).run(
        [photos], tmp_path / "a"
    )
    offline = build_services(settings, offline=True)
    pack = await Pipeline(offline, settings).run([photos], tmp_path / "b")
    assert all(call.cached for item in pack.items for call in item.calls)


async def test_offline_without_cache_fails_clearly(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    with pytest.raises(OfflineCacheMissError):
        await Pipeline(build_services(settings, offline=True), settings).run(
            [photos], tmp_path / "out"
        )


async def test_no_inputs_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="no supported"):
        await Pipeline(_services(FakeMetaClient([]), settings), settings).run([tmp_path], tmp_path)


def test_video_keyframes(tmp_path: Path) -> None:
    path = tmp_path / "pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(50):
        writer.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    writer.release()
    frames = stills(path, max_side=1280, max_frames=3)
    assert 1 <= len(frames) <= 3
    assert all(f.frame_index is not None and f.timestamp_s is not None for f in frames)
    assert frames[0].image.size == (160, 120)
    assert frames[0].stem.startswith("pan-f")


def test_collect_and_reject_media(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    assert collect_inputs([tmp_path]) == [photo]
    with pytest.raises(UnsupportedMediaError):
        stills(tmp_path / "doc.pdf", max_side=512)


async def test_failed_product_shot_keeps_the_listing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    client = FakeMetaClient(["lamp"])

    async def broken(**_: object) -> object:
        raise APIConnectionError(request=httpx2.Request("POST", "https://api.meta.ai/v1"))

    client.images.edit = broken
    messages: list[str] = []
    pack = await Pipeline(_services(client, settings), settings, messages.append).run(
        [photos], tmp_path / "out"
    )
    assert pack.items[0].product_shots == []
    assert any("no product shot" in m for m in messages)


def test_spark_open_routes_analysis_to_partner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("META_API_KEY", "meta")
    monkeypatch.setenv("BASETEN_API_KEY", "partner")
    monkeypatch.setenv("OMS_USE_SPARK_OPEN", "true")
    services = build_services(_settings(tmp_path))
    assert services.spark.model == "meta-models/muse-spark-open-1026"
    assert services.spark.client is not None
    assert "baseten" in str(services.spark.client.base_url)
    assert services.sam.client is not None
    assert "meta.ai" in str(services.sam.client.base_url)


def test_cli_offline_reports_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _photo(tmp_path / "photos")
    with pytest.raises(OfflineCacheMissError):
        main(["run", "photos", "--out", "pack", "--offline"])
