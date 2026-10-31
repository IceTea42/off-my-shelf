"""End-to-end S1 pipeline on a fake Meta client: media in, validated pack out."""

from dataclasses import replace
from pathlib import Path

import cv2
import httpx2
import numpy as np
import pytest
from openai import APIConnectionError, InternalServerError
from PIL import Image

from fakes import SAM_SIZE, SHOT_BAD, SHOT_OK, FakeMetaClient, fake_services
from off_my_shelf import pipeline
from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.cli import main
from off_my_shelf.config import Settings
from off_my_shelf.media import UnsupportedMediaError, collect_inputs, stills
from off_my_shelf.mocks import load_pack
from off_my_shelf.pipeline import Pipeline, Services, build_services, fallback_phrases, slugify
from off_my_shelf.schema import Language
from off_my_shelf.spark import ListingChain, ListingProvider, SceneItem, SparkAnalyzer


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, cache_dir=tmp_path / "cache", min_item_side_px=1, min_item_pixels=1
    )  # type: ignore[call-arg]


def _services(client: FakeMetaClient, settings: Settings) -> Services:
    return fake_services(client, DiskCache(settings.cache_dir), photographer=True)


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
    assert item.product_shots == ["shots/item-001-lamp.jpg"]  # mask barely occluded: no AI
    for rel in (item.detection.crop_path, item.detection.mask_path, *item.product_shots):
        assert rel is not None and (out / rel).is_file()
    assert (out / pack.media[0]).is_file()
    steps = [c.step.value for c in item.calls]
    assert steps == ["analyze", "segment", "analyze"]
    assert client.image_calls == []
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


@pytest.mark.parametrize("fmt", ["AVIF", "JXL", "HEIF", "WEBP", "TIFF", "BMP", "GIF", "PNG"])
def test_image_formats_load(tmp_path: Path, fmt: str) -> None:
    suffix = {"JXL": ".jxl", "HEIF": ".heic", "TIFF": ".tif"}.get(fmt, f".{fmt.lower()}")
    path = tmp_path / f"photo{suffix}"
    Image.new("RGB", (64, 48), (180, 40, 40)).save(path, fmt)
    assert collect_inputs([tmp_path]) == [path]
    (still,) = stills(path, max_side=1280)
    assert still.image.size == (64, 48) and still.image.mode == "RGB"


@pytest.mark.parametrize(("suffix", "codec"), [(".mkv", "mp4v"), (".avi", "MJPG")])
def test_video_containers_load(tmp_path: Path, suffix: str, codec: str) -> None:
    path = tmp_path / f"clip{suffix}"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), 10.0, (64, 48))
    rng = np.random.default_rng(1)
    for _ in range(20):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()
    frames = stills(path, max_side=1280, max_frames=2)
    assert frames and frames[0].image.size == (64, 48)


def test_collect_and_reject_media(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    assert collect_inputs([tmp_path]) == [photo]
    with pytest.raises(UnsupportedMediaError):
        stills(tmp_path / "doc.pdf", max_side=512)


async def test_failed_product_shot_keeps_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "HIDDEN_MIN_SHARE", 0.0)
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
    assert pack.items[0].product_shots == ["shots/item-001-lamp.jpg"]
    assert any(
        "real-pixel shot only (completion rejected: APIConnectionError)" in m for m in messages
    )


async def test_rejected_shots_are_retried_then_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "HIDDEN_MIN_SHARE", 0.0)
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    client = FakeMetaClient(["plate"])
    client.shot_checks = [SHOT_BAD, SHOT_BAD]
    messages: list[str] = []
    pack = await Pipeline(_services(client, settings), settings, messages.append).run(
        [photos], tmp_path / "out"
    )
    assert len(client.image_calls) == 2
    assert client.image_calls[1]["prompt"].startswith("A previous attempt was rejected: dent")
    assert pack.items[0].product_shots == ["shots/item-001-plate.jpg"]
    assert any("completion rejected: dent" in m for m in messages)


async def test_second_shot_attempt_can_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "HIDDEN_MIN_SHARE", 0.0)
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    client = FakeMetaClient(["plate"])
    client.shot_checks = [SHOT_BAD, SHOT_OK]
    pack = await Pipeline(_services(client, settings), settings).run([photos], tmp_path / "out")
    assert pack.items[0].product_shots == [
        "shots/item-001-plate-completed.jpg",
        "shots/item-001-plate.jpg",
    ]
    assert pack.items[0].detection.crop_path == "crops/item-001-plate.jpg"
    assert (tmp_path / "out" / "inputs" / "item-001-plate-to-fill.png").is_file()


async def test_redrawn_completion_is_rejected_by_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "HIDDEN_MIN_SHARE", 0.0)
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    client = FakeMetaClient(["vase"])
    client.redraw = True
    messages: list[str] = []
    pack = await Pipeline(_services(client, settings), settings, messages.append).run(
        [photos], tmp_path / "out"
    )
    assert len(client.image_calls) == 2
    assert pack.items[0].product_shots == ["shots/item-001-vase.jpg"]
    assert any("moved or redrawn" in m for m in messages)
    assert not any(
        c["response_format"]["json_schema"]["name"] == "ShotCheck" for c in client.chat_calls
    )


def test_slugify() -> None:
    assert slugify("Cake Stand!") == "cake-stand"
    assert slugify("black cast iron teapot") == "iron-teapot"
    assert slugify("Crème brûlée dish") == "brulee-dish"
    assert slugify("!!!") == "item"


def test_listings_go_to_partners_first_then_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("META_API_KEY", "meta")
    monkeypatch.setenv("BASETEN_API_KEY", "partner")
    monkeypatch.setenv("MODAL_SPARK_OPEN_URL", "https://me--spark-open.modal.run/v1")
    monkeypatch.setenv("MODAL_SPARK_OPEN_KEY", "modal")
    services = build_services(_settings(tmp_path))
    providers = services.listing.providers
    assert [p.name for p in providers] == ["baseten", "modal", "meta"]
    assert [p.analyzer.model for p in providers[:2]] == ["meta-models/muse-spark-open-1026"] * 2
    assert "baseten" in str(providers[0].analyzer.client.base_url)  # type: ignore[union-attr]
    assert "modal.run" in str(providers[1].analyzer.client.base_url)  # type: ignore[union-attr]
    assert providers[2].analyzer is services.spark
    assert services.sam.client is not None and "meta.ai" in str(services.sam.client.base_url)


async def test_listing_falls_back_when_a_partner_fails(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _photo(tmp_path / "photos")
    cache = DiskCache(settings.cache_dir)
    services = fake_services(FakeMetaClient(["lamp"]), cache, photographer=True)
    down = httpx2.Response(503, request=httpx2.Request("POST", "https://inference.baseten.co"))
    broken = FakeMetaClient(["lamp"])

    async def unavailable(**_: object) -> object:
        raise InternalServerError("down", response=down, body=None)

    broken.chat.completions.create = unavailable
    partner = SparkAnalyzer(broken, "meta-models/muse-spark-open-1026", cache, 4000)  # type: ignore[arg-type]
    chain = ListingChain([ListingProvider("baseten", partner), *services.listing.providers])
    messages: list[str] = []
    pack = await Pipeline(replace(services, listing=chain), settings, messages.append).run(
        [tmp_path / "photos"], tmp_path / "out"
    )

    listing_call = next(
        c for c in pack.items[0].calls if c.model == "muse-spark-1.3" and c.provider is None
    )
    assert listing_call is not None
    assert any("fell back past baseten (InternalServerError)" in m for m in messages)


def test_cli_offline_reports_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _photo(tmp_path / "photos")
    with pytest.raises(OfflineCacheMissError):
        main(["run", "photos", "--out", "pack", "--offline"])


async def test_too_small_items_and_photos_are_skipped_with_warnings(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    photos = tmp_path / "photos"
    _photo(photos)
    Image.new("RGB", (200, 150), "white").save(photos / "thumb.png")
    client = FakeMetaClient(phrases=["mug"])

    pack = await Pipeline(_services(client, settings), settings).run([photos], tmp_path / "out")

    assert pack.items == []
    assert client.chat_calls[-1]["response_format"]["json_schema"]["name"] == "SceneInventory"
    small_photo, small_item = sorted(pack.warnings, key=lambda w: "thumb" not in w)
    assert "thumb.png: 200x150 px is too small" in small_photo
    assert "'mug' covers only" in small_item


def test_fallback_phrases() -> None:
    item = SceneItem(phrase="black crossbody bag", label="Bag", short_name="handbag")
    assert fallback_phrases(item) == ["black crossbody bag", "handbag", "bag"]
    item = SceneItem(phrase="mug", label="Mug", short_name="mug")
    assert fallback_phrases(item) == ["mug"]


async def test_missed_phrase_is_retried_with_simpler_names(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeMetaClient(phrases=["black crossbody bag"])
    client.sam_misses = {"black crossbody bag"}  # found by its short name instead
    pack = await Pipeline(_services(client, settings), settings).run(
        [_photo(tmp_path / "in")], tmp_path / "out"
    )
    asked = [call["input"][0]["content"][0]["text"] for call in client.sam_calls]
    assert asked == ["black crossbody bag", "bag"]
    assert len(pack.items) == 1


async def test_unfound_item_is_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeMetaClient(phrases=["blue bag", "mug"])
    client.sam_misses = {"blue bag", "bag"}
    pack = await Pipeline(_services(client, settings), settings).run(
        [_photo(tmp_path / "in")], tmp_path / "out"
    )
    assert any("no mask for 'blue bag'" in warning for warning in pack.warnings)


async def test_all_photos_too_small_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    Image.new("RGB", (100, 100), "white").save(tmp_path / "tiny.png")
    with pytest.raises(ValueError, match="too small"):
        await Pipeline(_services(FakeMetaClient(["mug"]), settings), settings).run(
            [tmp_path / "tiny.png"], tmp_path / "out"
        )


async def test_video_item_is_listed_once_from_its_best_frame(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "shelf.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, SAM_SIZE)
    rng = np.random.default_rng(1)
    for _ in range(40):  # 4 s of noise: several sharp, well-spaced keyframes
        writer.write(rng.integers(0, 255, (SAM_SIZE[1], SAM_SIZE[0], 3), dtype=np.uint8))
    writer.release()
    client = FakeMetaClient(phrases=["lamp"])

    pack = await Pipeline(_services(client, settings), settings).run([path], tmp_path / "out")

    names = [c["response_format"]["json_schema"]["name"] for c in client.chat_calls]
    assert names.count("SceneInventory") == len(pack.media) > 1
    assert names.count("ItemAnalysis") == len(pack.items) == 1
    later_frame = next(
        c
        for c in client.chat_calls[1:]
        if c["response_format"]["json_schema"]["name"] == "SceneInventory"
    )
    assert "Items listed from earlier frames: lamp." in str(later_frame["messages"])


async def test_cards_are_published_before_the_completed_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "HIDDEN_MIN_SHARE", 0.0)
    settings = _settings(tmp_path)
    photos = tmp_path / "photos"
    _photo(photos)
    _photo(photos, "second.png")  # two photos are scanned side by side
    client = FakeMetaClient(["plate"])
    services = _services(client, settings)
    run = Pipeline(services, settings)
    seen_while_completing: list[list[str]] = []
    complete = services.photographer.complete  # type: ignore[union-attr]

    async def watching(*args: object) -> object:
        seen_while_completing.append([item.product_shots[0] for item in run.partial.values()])
        return await complete(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(services.photographer, "complete", watching)
    pack = await run.run([photos], tmp_path / "out")

    assert [item.item_id for item in pack.items] == ["item-001", "item-002"]
    # Each card was listed with its real-pixel shot while its completion was still running.
    assert seen_while_completing[0] == ["shots/item-001-plate.jpg"]
    assert "shots/item-002-plate.jpg" in seen_while_completing[1]
    assert list(run.partial.values()) == pack.items
    assert run.items_done == run.items_total == 2
