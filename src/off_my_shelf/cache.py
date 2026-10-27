"""Content-addressed disk cache for model results.

Keys are SHA-256 hashes of the request (model, prompt, input bytes), so the same input never
pays twice and demos replay offline. Writes are atomic (temp file + replace), which is safe on
Windows and Linux alike.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class OfflineCacheMissError(LookupError):
    """A model result is needed, no API client is configured and the cache has no entry."""

    def __init__(self, step: str, key: str) -> None:
        super().__init__(f"offline mode: no cached {step} result for {key}")


def cache_key(namespace: str, *parts: JsonValue | bytes) -> str:
    """Stable hash of a namespace plus request parts (JSON-able values or raw bytes)."""
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for part in parts:
        if isinstance(part, bytes):
            digest.update(b"\x00b")
            digest.update(hashlib.sha256(part).digest())
        else:
            digest.update(b"\x00j")
            digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return f"{namespace}-{digest.hexdigest()[:32]}"


class DiskCache:
    """Stores JSON documents and binary blobs under ``root``, one file per key."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str, suffix: str) -> Path:
        namespace, _, digest = key.rpartition("-")
        return self.root / namespace / digest[:2] / f"{digest}{suffix}"

    def get_json(self, key: str) -> JsonValue:
        """The stored document, or None when the key is absent."""
        path = self._path(key, ".json")
        if not path.is_file():
            return None
        value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
        return value

    def put_json(self, key: str, value: JsonValue) -> None:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        self._write(self._path(key, ".json"), text.encode("utf-8"))

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key, ".bin")
        return path.read_bytes() if path.is_file() else None

    def put_bytes(self, key: str, value: bytes) -> None:
        self._write(self._path(key, ".bin"), value)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
