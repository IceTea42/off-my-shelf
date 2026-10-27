"""Muse Voice Transcribe: spoken notes about an item, as text.

The API takes 16-bit mono PCM WAV at 16 or 24 kHz, at most 10 minutes. ``to_api_wav`` converts
any PCM WAV (other rates, stereo, 8/24/32-bit) with the standard library and NumPy only, so no
ffmpeg install is needed on Windows or Linux.
"""

import io
import json
import time
import wave
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import transcribe_cost
from off_my_shelf.schema import ModelCall, PipelineStep

API_RATE = 16_000
MAX_SECONDS = 600
LANGUAGE_BIAS = ["English", "French", "Turkish"]
UNSIGNED_8BIT, PACKED_24BIT = 1, 3


class AudioFormatError(ValueError):
    """The audio is not a PCM WAV file the converter understands."""


def to_api_wav(data: bytes) -> bytes:
    """Any PCM WAV -> 16 kHz mono 16-bit WAV, trimmed to the API's 10-minute limit."""
    try:
        with wave.open(io.BytesIO(data)) as source:
            channels, width = source.getnchannels(), source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as error:
        raise AudioFormatError(f"not a PCM WAV file: {error}") from error
    if width not in {1, 2, 3, 4}:
        raise AudioFormatError(f"unsupported sample width: {width} bytes")

    if width == PACKED_24BIT:  # widen to 32-bit little-endian
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        samples = (np.pad(raw, ((0, 0), (1, 0))).view("<i4").ravel() >> 8).astype(np.float64)
        scale = 2.0**23
    elif width == UNSIGNED_8BIT:  # 8-bit WAV is unsigned
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0
        scale = 128.0
    else:
        samples = np.frombuffer(frames, dtype=f"<i{width}").astype(np.float64)
        scale = 2.0 ** (8 * width - 1)
    mono = samples.reshape(-1, channels).mean(axis=1) / scale

    target_len = int(len(mono) * API_RATE / rate)
    if target_len != len(mono):
        mono = np.interp(np.linspace(0, len(mono) - 1, target_len), np.arange(len(mono)), mono)
    mono = mono[: API_RATE * MAX_SECONDS]
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2")

    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(API_RATE)
        target.writeframes(pcm.tobytes())
    return out.getvalue()


class Transcriber:
    """``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def transcribe(self, wav: bytes) -> tuple[str, ModelCall]:
        audio = to_api_wav(wav)
        request = {
            "model": self.model,
            "audioEncoding": "WAV",
            "mode": "PUSH_TO_TALK",
            "languageBias": LANGUAGE_BIAS,
        }
        key = cache_key("transcribe", request, audio)
        cached = self.cache.get_json(key)
        latency_ms: int | None = None
        if isinstance(cached, dict):
            reply: dict[str, Any] = cached
        elif self.client is None:
            raise OfflineCacheMissError(PipelineStep.TRANSCRIBE.value, key)
        else:
            start = time.perf_counter()
            # Custom multipart endpoint: the SDK's generic POST keeps auth, retries and timeouts.
            files: list[tuple[str, tuple[str | None, bytes, str]]] = [
                ("request", (None, json.dumps(request).encode("utf-8"), "application/json")),
                ("audio", ("note.wav", audio, "audio/wav")),
            ]
            reply = await self.client.post(
                "/asr/transcribe",
                cast_to=dict[str, Any],
                files=files,
                # The SDK swaps this for the real multipart boundary header.
                options={
                    "headers": {"Accept": "application/json", "Content-Type": "multipart/form-data"}
                },
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            self.cache.put_json(key, reply)
        duration_ms = int(reply.get("audioDurationMs") or len(audio) // 32)
        call = ModelCall(
            step=PipelineStep.TRANSCRIBE,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            cost_usd=transcribe_cost(duration_ms),
            latency_ms=latency_ms,
        )
        return str(reply.get("transcript", "")).strip(), call
