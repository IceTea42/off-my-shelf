"""Web app: config, demo run, upload run on a fake client, downloads and errors."""

import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
        _env_file=None, cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs"
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
        assert "Off My Shelf" in client.get("/").text
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
        pack = run["pack"]
        assert isinstance(pack, dict)
        assert pack["items"][0]["analysis"]["listing"]["language"] == "fr"
        assert pack["items"][0]["product_shots"] == ["shots/item-001.webp"]


def test_bad_requests(tmp_path: Path) -> None:
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        bad = client.post("/api/runs", files=[("files", ("notes.txt", b"hi", "text/plain"))])
        assert bad.status_code == 400
        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/runs/nope/zip").status_code == 404
