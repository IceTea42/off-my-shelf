"""My trials: ownership, hiding, reopening after a restart, rolling buffer, rate limit."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from fakes import FakeMetaClient, fake_services
from off_my_shelf import trials, web
from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings
from off_my_shelf.trials import RateLimiter, Trial, all_trials, prune, save_trial
from test_web import _png, _settings, _wait


def test_prune_keeps_a_rolling_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trials, "MAX_RUNS", 2)
    now = datetime(2026, 11, 1, tzinfo=UTC)
    for n, age_days in enumerate([0, 1, 2, 40]):
        created = now - timedelta(days=age_days)
        save_trial(tmp_path, Trial(run_id=f"{n:012x}", visitor="v1234567", created_at=created))
    assert sorted(prune(tmp_path, now)) == ["000000000002", "000000000003"]
    assert [t.run_id for t in all_trials(tmp_path)] == ["000000000000", "000000000001"]


def test_rate_limiter_rolls_over_the_hour() -> None:
    limiter = RateLimiter(limit=2)
    assert limiter.allow("a", 0) and limiter.allow("a", 10) and not limiter.allow("a", 20)
    assert limiter.allow("b", 20)
    assert limiter.allow("a", 3700)


def test_my_trials_hide_and_reopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings: Settings = _settings(tmp_path).model_copy(
        update={"meta_api_key": SecretStr("test-key")}
    )
    fake = FakeMetaClient(["lamp"])
    cache = DiskCache(settings.cache_dir)
    monkeypatch.setattr(web, "build_services", lambda *_, **__: fake_services(fake, cache))
    upload = [("files", ("shelf.png", _png(), "image/png"))]
    with TestClient(web.create_app(settings)) as client:
        client.cookies.set("oms_visitor", "visitor-one")
        run_id = client.post("/api/runs", files=upload).json()["run_id"]
        assert _wait(client, run_id)["status"] == "done"
        [row] = client.get("/api/trials").json()
        assert row["run_id"] == run_id and row["items"] == 1
        assert row["thumbnail"] == "media/upload-01.jpg"

        client.cookies.set("oms_visitor", "visitor-two")
        assert client.get("/api/trials").json() == []  # nobody sees another browser's runs
        assert client.post(f"/api/trials/{run_id}/hide").status_code == 404

    with TestClient(web.create_app(settings)) as client:  # a restart: memory is empty
        client.cookies.set("oms_visitor", "visitor-one")
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "done"
        assert client.post(f"/api/trials/{run_id}/hide").json() == {"hidden": True}
        assert client.get("/api/trials").json() == []
        assert (settings.runs_dir / run_id / "pack").is_dir()  # hidden, still in the buffer


def test_live_runs_are_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path).model_copy(update={"meta_api_key": SecretStr("test-key")})
    fake = FakeMetaClient(["lamp"])
    cache = DiskCache(settings.cache_dir)
    monkeypatch.setattr(web, "build_services", lambda *_, **__: fake_services(fake, cache))
    monkeypatch.setattr(web, "RateLimiter", lambda: RateLimiter(limit=1))
    upload = [("files", ("shelf.png", _png(), "image/png"))]
    with TestClient(web.create_app(settings)) as client:
        assert client.post("/api/runs", files=upload).status_code == 202
        assert client.post("/api/runs", files=upload).status_code == 429
