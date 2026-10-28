"""Web app: config, demo run, upload run on a fake client, downloads and errors."""

import io
import time
import wave
import zipfile
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import InternalServerError
from PIL import Image
from pydantic import SecretStr

from fakes import SAM_SIZE, FakeMetaClient
from off_my_shelf import web
from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings
from off_my_shelf.pipeline import Services
from off_my_shelf.product_shots import ProductPhotographer
from off_my_shelf.sam import SamSegmenter
from off_my_shelf.spark import SparkAnalyzer


def _settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        min_item_side_px=1,
        min_item_pixels=1,
    )


def _wait(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(100):
        run: dict[str, object] = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", SAM_SIZE, "beige").save(buffer, "PNG")
    return buffer.getvalue()


def test_page_config_and_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        assert 'href="/favicon.svg"' in client.get("/").text
        assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg+xml")
        assert client.get("/favicon.ico").headers["content-type"] == "image/png"
        config = client.get("/api/config").json()
        assert config["offline"] is True and config["default_language"] == "en"

        run_id = client.post("/api/demo").json()["run_id"]
        run = _wait(client, run_id)
        assert run["status"] == "done"
        pack = run["pack"]
        assert isinstance(pack, dict) and len(pack["items"]) == 5
        crop = pack["items"][0]["detection"]["crop_path"]
        assert client.get(f"{run['base_url']}{crop}").status_code == 200

        archive = client.get(f"/api/runs/{run_id}/zip")
        assert archive.status_code == 200
        assert "index.html" in zipfile.ZipFile(io.BytesIO(archive.content)).namelist()


def test_upload_runs_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    fake = FakeMetaClient(["lamp"])
    cache = DiskCache(settings.cache_dir)

    def fake_services(*_: object, **__: object) -> Services:
        return Services(
            sam=SamSegmenter(fake, "sam-3.1", cache),  # type: ignore[arg-type]
            spark=SparkAnalyzer(fake, "muse-spark-1.3", cache, 4000),  # type: ignore[arg-type]
            photographer=ProductPhotographer(fake, "muse-image-1.0", cache),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(web, "build_services", fake_services)
    with TestClient(web.create_app(settings)) as client:
        response = client.post(
            "/api/runs",
            files=[("files", ("shelf.png", _png(), "image/png"))],
            data={"language": "fr", "product_shots": "true"},
        )
        assert response.status_code == 202
        run = _wait(client, response.json()["run_id"])
        assert run["status"] == "done", run["error"]
        assert run["items_done"] == run["items_total"] == 1
        count = run["message_count"]
        assert isinstance(count, int) and count > 1
        tail = client.get(f"/api/runs/{response.json()['run_id']}?since={count - 1}").json()
        assert tail["messages"] == ["done: 1 items"]
        pack = run["pack"]
        assert isinstance(pack, dict)
        assert pack["items"][0]["analysis"]["listing"]["language"] == "fr"
        assert pack["items"][0]["product_shots"] == ["shots/item-001-lamp.jpg"]


def test_bad_requests(tmp_path: Path) -> None:
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        bad = client.post("/api/runs", files=[("files", ("notes.txt", b"hi", "text/plain"))])
        assert bad.status_code == 400
        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/runs/nope/zip").status_code == 404


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(b"\x10\x00" * 16_000)
    return buffer.getvalue()


def test_transcribe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMetaClient(["lamp"])
    monkeypatch.setattr(web, "make_client", lambda _settings: fake)
    settings = _settings(tmp_path).model_copy(update={"meta_api_key": SecretStr("test-key")})
    with TestClient(web.create_app(settings)) as client:
        voice = [("voice", ("goal.wav", _wav(), "audio/wav"))]
        reply = client.post("/api/transcribe", files=voice)
        assert reply.status_code == 200
        assert reply.json()["transcript"] == fake.transcript
        assert (
            client.post(
                "/api/transcribe", files=[("voice", ("x.wav", b"no", "audio/wav"))]
            ).status_code
            == 400
        )
        fake.transcript = " "
        silent = [("voice", ("other.wav", _wav()[:-2] + b"\x20\x00", "audio/wav"))]
        assert client.post("/api/transcribe", files=silent).status_code == 422


def test_transcribe_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        voice = [("voice", ("goal.wav", _wav(), "audio/wav"))]
        assert client.post("/api/transcribe", files=voice).status_code == 503


def test_budget_stops_live_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "make_client", lambda _settings: FakeMetaClient(["lamp"]))
    settings = _settings(tmp_path).model_copy(
        update={"meta_api_key": SecretStr("test-key"), "budget_usd": 0.00001}
    )
    with TestClient(web.create_app(settings)) as client:
        voice = [("voice", ("goal.wav", _wav(), "audio/wav"))]
        assert client.post("/api/transcribe", files=voice).status_code == 200
        assert client.post("/api/transcribe", files=voice).status_code == 429
        upload = [("files", ("shelf.png", _png(), "image/png"))]
        assert client.post("/api/runs", files=upload).status_code == 429


def test_transcribe_debug_keeps_notes_and_reports_api_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMetaClient([])
    monkeypatch.setattr(web, "make_client", lambda _settings: fake)
    settings = _settings(tmp_path).model_copy(
        update={"meta_api_key": SecretStr("test-key"), "debug_voice": True}
    )
    url = "https://api.meta.ai/v1/asr/transcribe"
    response = httpx2.Response(
        502, request=httpx2.Request("POST", url), json={"error": {"type": "upstream"}}
    )
    fake.post_error = InternalServerError(
        "bad gateway", response=response, body={"type": "upstream"}
    )
    with TestClient(web.create_app(settings)) as client:
        reply = client.post("/api/transcribe", files=[("voice", ("goal.wav", _wav(), "audio/wav"))])
    assert reply.status_code == 502
    assert "HTTP 502 from POST /v1/asr/transcribe: {'type': 'upstream'}" in reply.json()["detail"]
    kept = sorted(path.name.split("-", 2)[-1] for path in (settings.cache_dir / "voice").iterdir())
    assert kept == ["api.wav", "result.txt", "upload.wav"]


def test_installable_app_files(tmp_path: Path) -> None:
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        page = client.get("/").text
        assert 'rel="manifest"' in page and 'capture="environment"' in page
        manifest = client.get("/manifest.webmanifest")
        assert manifest.headers["content-type"].startswith("application/manifest+json")
        icons = manifest.json()["icons"]
        for icon in icons:
            assert client.get(icon["src"]).headers["content-type"] == "image/png"
        assert {icon["sizes"] for icon in icons} >= {"192x192", "512x512"}
        worker = client.get("/sw.js")
        assert worker.headers["content-type"].startswith("text/javascript")
        assert worker.headers["cache-control"] == "no-cache"


def test_share_target_starts_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    fake = FakeMetaClient(["lamp"])
    cache = DiskCache(settings.cache_dir)
    monkeypatch.setattr(
        web,
        "build_services",
        lambda *_, **__: Services(
            sam=SamSegmenter(fake, "sam-3.1", cache),  # type: ignore[arg-type]
            spark=SparkAnalyzer(fake, "muse-spark-1.3", cache, 4000),  # type: ignore[arg-type]
        ),
    )
    with TestClient(web.create_app(settings)) as client:
        shared = client.post(
            "/share", files=[("files", ("IMG_0042", _png(), "image/png"))], follow_redirects=False
        )
        assert shared.status_code == 303
        run_id = shared.headers["location"].removeprefix("/?run=")
        assert _wait(client, run_id)["status"] == "done"


def test_translate_once_then_from_the_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMetaClient([])
    monkeypatch.setattr(
        web,
        "build_services",
        lambda *_, **__: Services(
            sam=SamSegmenter(fake, "sam-3.1", DiskCache(tmp_path)),  # type: ignore[arg-type]
            spark=SparkAnalyzer(fake, "muse-spark-1.3", DiskCache(tmp_path), 4000),  # type: ignore[arg-type]
        ),
    )
    settings = _settings(tmp_path).model_copy(update={"meta_api_key": SecretStr("test-key")})
    with TestClient(web.create_app(settings)) as client:
        run_id = client.post("/api/demo").json()["run_id"]
        url = f"/api/runs/{run_id}/items/item-001/translate"
        english = client.post(url, data={"language": "en"}).json()
        assert english["language"] == "en" and english["title"].startswith("Black articulated")
        assert client.post(url, data={"language": "en"}).json() == english
        assert client.post(url, data={"language": "fr"}).json()["title"].startswith("Lampe")
        assert len(fake.chat_calls) == 1
        item = client.get(f"/api/runs/{run_id}").json()["pack"]["items"][0]
        assert [t["language"] for t in item["translations"]] == ["en"]
        assert (
            client.post(
                f"/api/runs/{run_id}/items/nope/translate", data={"language": "en"}
            ).status_code
            == 404
        )
