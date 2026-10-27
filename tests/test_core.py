"""Settings, disk cache, cost model and client construction."""

from pathlib import Path

import pytest

from off_my_shelf.cache import DiskCache, cache_key
from off_my_shelf.config import Settings
from off_my_shelf.costs import token_cost, transcribe_cost
from off_my_shelf.meta_client import MissingApiKeyError, make_client
from off_my_shelf.schema import Language


def _settings() -> Settings:
    """Settings from the process environment only, ignoring any local .env file."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    settings = _settings()
    assert settings.meta_api_key is None
    assert settings.default_language is Language.EN
    assert settings.spark_model == "muse-spark-1.3"


@pytest.mark.parametrize("name", ["META_API_KEY", "MODEL_API_KEY"])
def test_api_key_read_from_either_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv(name, "secret-value")
    settings = _settings()
    assert settings.meta_api_key is not None
    assert settings.meta_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_cache_key_is_stable_and_order_insensitive_for_dicts() -> None:
    assert cache_key("spark", {"a": 1, "b": 2}) == cache_key("spark", {"b": 2, "a": 1})
    assert cache_key("spark", {"a": 1}) != cache_key("sam", {"a": 1})
    assert cache_key("x", b"abc") != cache_key("x", "abc")


def test_disk_cache_roundtrip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = cache_key("spark", {"prompt": "hi"})
    assert cache.get_json(key) is None
    cache.put_json(key, {"answer": "élan"})
    assert cache.get_json(key) == {"answer": "élan"}
    cache.put_bytes(key, b"\x89PNG")
    assert cache.get_bytes(key) == b"\x89PNG"
    assert not list(tmp_path.rglob(".tmp-*"))


def test_costs() -> None:
    assert token_cost("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.50)
    assert token_cost("unknown-model", 10, 10) is None
    assert transcribe_cost(3_600_000) == pytest.approx(0.18)


def test_make_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        make_client(_settings())


def test_make_client_points_at_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "k")
    client = make_client(_settings())
    assert str(client.base_url).startswith("https://api.meta.ai/v1")
