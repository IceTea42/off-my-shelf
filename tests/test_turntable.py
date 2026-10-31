"""3D turn: the request sent to the SAM 3D endpoint, saved files and cost, web wiring."""

import base64
import io
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from off_my_shelf import turntable, web
from off_my_shelf.config import Settings
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack
from off_my_shelf.schema import PipelineStep


def _settings(tmp_path: Path, url: str | None = "https://sam3d.test/turn") -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        meta_api_key=SecretStr("m"),
        modal_sam3d_url=url,
        modal_sam3d_key=SecretStr("k") if url else None,
    )


def _fake_endpoint(monkeypatch: pytest.MonkeyPatch, status: int = 200) -> list[dict[str, Any]]:
    """Answer like scripts/modal_sam3d.py; returns the requests it received."""
    received: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        received.append({"auth": request.headers["authorization"], **json.loads(request.content)})
        body = {
            "video": base64.b64encode(b"mp4").decode(),
            "model": base64.b64encode(b"glb").decode(),
        }
        return httpx.Response(status, json=body if status == 200 else {"detail": "boom"})

    real: Callable[..., httpx.AsyncClient] = httpx.AsyncClient
    monkeypatch.setattr(
        turntable.httpx,  # type: ignore[attr-defined]
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(answer), **kwargs),
    )
    return received


async def test_turn_sends_crop_and_matching_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    item = load_mock_pack().items[0]
    received = _fake_endpoint(monkeypatch)

    call = await turntable.make_turntable(_settings(tmp_path), pack_dir, item)

    sent = received[0]
    assert sent["auth"] == "Bearer k" and sent["seconds"] == turntable.SECONDS
    photo = Image.open(io.BytesIO(base64.b64decode(sent["image"])))
    mask = Image.open(io.BytesIO(base64.b64decode(sent["mask"])))
    assert photo.size == mask.size and mask.getextrema()[1] == 255
    assert (pack_dir / "turns/item-001.mp4").read_bytes() == b"mp4"
    assert (pack_dir / "models/item-001.glb").read_bytes() == b"glb"
    assert item.turntable_video == "turns/item-001.mp4"
    assert call.step is PipelineStep.TURNTABLE and call.provider == "modal"


async def test_turn_needs_the_endpoint(tmp_path: Path) -> None:
    with pytest.raises(turntable.TurntableError, match="not set up"):
        await turntable.make_turntable(
            _settings(tmp_path, url=None), MOCK_PACK_DIR, load_mock_pack().items[0]
        )


def test_web_turn_once_then_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received = _fake_endpoint(monkeypatch)
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        assert client.get("/api/config").json()["turntable"] is True
        run_id = client.post("/api/demo").json()["run_id"]
        first = client.post(f"/api/runs/{run_id}/items/item-002/turn").json()
        assert first == {"video": "turns/item-002.mp4", "model": "models/item-002.glb"}
        assert client.post(f"/api/runs/{run_id}/items/item-002/turn").json() == first
        assert len(received) == 1
        assert client.get(f"/runs/{run_id}/pack/turns/item-002.mp4").content == b"mp4"


def test_web_turn_reports_endpoint_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_endpoint(monkeypatch, status=500)
    with TestClient(web.create_app(_settings(tmp_path))) as client:
        run_id = client.post("/api/demo").json()["run_id"]
        response = client.post(f"/api/runs/{run_id}/items/item-002/turn")
        assert response.status_code == 502 and "500" in response.json()["detail"]
