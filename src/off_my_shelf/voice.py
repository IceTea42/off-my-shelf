"""Muse Voice Transcribe: spoken goals and notes, as text.

The API takes 16-bit mono PCM WAV at 16 or 24 kHz, at most 10 minutes. ``to_api_wav`` decodes
any common recording (browser WebM/Opus or Ogg, M4A, MP3, any WAV) with PyAV, whose wheels bundle
FFmpeg, so nothing else needs installing on Windows or Linux.
"""

import io
import json
import time
import wave
from typing import Any

import av
import numpy as np
from openai import AsyncOpenAI

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import transcribe_cost
from off_my_shelf.schema import ModelCall, PipelineStep

API_RATE = 16_000
MAX_SECONDS = 600


class AudioFormatError(ValueError):
    """The upload is not audio PyAV can decode, or holds no sound."""


def to_api_wav(data: bytes) -> bytes:
    """Any recording -> 16 kHz mono 16-bit WAV, trimmed to the API's 10-minute limit."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=API_RATE)
    chunks: list[np.ndarray[Any, Any]] = []  # s16 frames
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                raise AudioFormatError("the file has no audio track")
            for frame in container.decode(container.streams.audio[0]):
                chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    except av.FFmpegError as error:
        raise AudioFormatError(f"cannot decode the audio: {error}") from error
    pcm = np.concatenate([c.reshape(-1) for c in chunks]) if chunks else np.zeros(0, "<i2")
    if pcm.size == 0:
        raise AudioFormatError("the recording holds no sound")
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(API_RATE)
        target.writeframes(pcm[: API_RATE * MAX_SECONDS].astype("<i2").tobytes())
    return out.getvalue()


def wav_seconds(wav: bytes) -> float:
    """Length of a WAV produced by ``to_api_wav``."""
    with wave.open(io.BytesIO(wav)) as source:
        return float(source.getnframes() / source.getframerate())


class Transcriber:
    """``client=None`` means cache-only (offline)."""

    def __init__(self, client: AsyncOpenAI | None, model: str, cache: DiskCache) -> None:
        self.client = client
        self.model = model
        self.cache = cache

    async def transcribe(self, recording: bytes) -> tuple[str, ModelCall]:
        audio = to_api_wav(recording)
        request = {
            "model": self.model,
            "audioEncoding": "WAV",
            "mode": "PUSH_TO_TALK",  # language auto-detected (25 incl. English, French, Turkish)
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
