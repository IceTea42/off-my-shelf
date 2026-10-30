"""One cheap live call per model, to check the key, the endpoint and the response shape.

Usage:
    uv run python -m off_my_shelf.smoke spark
    uv run python -m off_my_shelf.smoke sam --image path/to/photo.jpg --phrase mug
    uv run python -m off_my_shelf.smoke vision --image path/to/photo.jpg
    uv run python -m off_my_shelf.smoke image --image path/to/photo.jpg --phrase teapot
    uv run python -m off_my_shelf.smoke open --image path/to/photo.jpg   # Muse Spark Open
    uv run python -m off_my_shelf.smoke transcribe [--audio note.wav]  # default: last kept web note
"""

import argparse
import asyncio
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import numpy as np

from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings, get_settings
from off_my_shelf.costs import token_cost
from off_my_shelf.imaging import PixelBox, encode, load_image, pixel_lock, studio_canvas
from off_my_shelf.meta_client import (
    make_client,
    make_open_weights_client,
    open_weight_endpoints,
)
from off_my_shelf.mocks import MOCK_PACK_DIR
from off_my_shelf.product_shots import ProductPhotographer
from off_my_shelf.sam import SamSegmenter
from off_my_shelf.spark import SparkAnalyzer
from off_my_shelf.voice import Transcriber

DEFAULT_IMAGE = MOCK_PACK_DIR / "media" / "shelf_01.png"


async def smoke_spark(settings: Settings, _: argparse.Namespace) -> None:
    client = make_client(settings)
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=settings.spark_model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        # Reasoning tokens count toward this budget: leave room for thinking and the answer.
        max_tokens=512,
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    text = response.choices[0].message.content
    usage = response.usage
    print(f"model={settings.spark_model} reply={text!r} latency_ms={latency_ms}")
    if usage is not None:
        cost = token_cost(settings.spark_model, usage.prompt_tokens, usage.completion_tokens)
        details = usage.completion_tokens_details
        reasoning = details.reasoning_tokens if details is not None else None
        print(
            f"tokens in={usage.prompt_tokens} out={usage.completion_tokens} "
            f"(reasoning={reasoning}) cost_usd={cost}"
        )


async def smoke_sam(settings: Settings, args: argparse.Namespace) -> None:
    image = load_image(args.image, settings.max_image_side_px)
    cache = DiskCache(settings.cache_dir)
    segmenter = SamSegmenter(make_client(settings), settings.sam_model, cache)
    run = await segmenter.segment(image, args.phrase)
    print(f"model={settings.sam_model} phrase={args.phrase!r} image={image.size}")
    print(f"objects={len(run.segments)} cached={run.call.cached} latency_ms={run.call.latency_ms}")
    for segment in run.segments:
        print(
            f"  id={segment.object_id} box={segment.box} conf={segment.confidence} "
            f"mask_px={int(segment.mask.sum())}"
        )


async def smoke_vision(settings: Settings, args: argparse.Namespace) -> None:
    """Scene inventory, then a full listing for the whole photo, with structured output."""
    image = load_image(args.image, settings.max_image_side_px)
    analyzer = SparkAnalyzer(
        make_client(settings),
        settings.spark_model,
        DiskCache(settings.cache_dir),
        settings.max_output_tokens,
        settings.reasoning_effort,
    )
    inventory = await analyzer.inventory(image)
    print(f"inventory: {[item.phrase for item in inventory.value.items]}")
    print(f"  tokens in={inventory.call.input_tokens} out={inventory.call.output_tokens}")
    listing = await analyzer.analyze(image, settings.default_language)
    print(listing.value.model_dump_json(indent=2))
    print(
        f"  tokens in={listing.call.input_tokens} out={listing.call.output_tokens} "
        f"cost_usd={listing.call.cost_usd} latency_ms={listing.call.latency_ms}"
    )


async def smoke_open(settings: Settings, args: argparse.Namespace) -> None:
    """One listing from Muse Spark Open at each configured partner (Baseten, Modal)."""
    endpoints = open_weight_endpoints(settings)
    if not endpoints:
        raise SystemExit("no partner configured: set BASETEN_API_KEY and/or MODAL_SPARK_OPEN_*")
    image = load_image(args.image, settings.max_image_side_px)
    for name, url, key in endpoints:
        analyzer = SparkAnalyzer(
            make_open_weights_client(settings, url, key),
            settings.spark_open_model,
            DiskCache(settings.cache_dir),
            settings.max_output_tokens,
        )
        listing = await analyzer.analyze(image, settings.default_language)
        print(f"{name}: model={settings.spark_open_model} at {url}")
        print(f"  title: {listing.value.listing.title!r} price: {listing.value.price.suggested}")
        print(
            f"  tokens in={listing.call.input_tokens} out={listing.call.output_tokens} "
            f"latency_ms={listing.call.latency_ms}"
        )


async def smoke_image(settings: Settings, args: argparse.Namespace) -> None:
    """Hide the centre of the photo, let Muse Image fill it, pixel-lock, save for a look."""
    image = load_image(args.image, settings.max_image_side_px)
    mask = np.ones((image.height, image.width), dtype=bool)
    h, w = image.height, image.width
    mask[h * 2 // 5 : h * 3 // 5, w * 2 // 5 : w * 3 // 5] = False
    canvas = studio_canvas(image, mask, PixelBox(0, 0, w, h))
    cache = DiskCache(settings.cache_dir)
    photographer = ProductPhotographer(make_client(settings), settings.image_model, cache)
    filled, call = await photographer.complete(canvas.to_fill, args.phrase, [])
    locked, drift = pixel_lock(filled, canvas)
    out = settings.cache_dir / "smoke_completed.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode(locked, "JPEG"))
    print(
        f"model={settings.image_model} drift={drift:.3f} cached={call.cached} "
        f"latency_ms={call.latency_ms} -> {out}"
    )


async def smoke_transcribe(settings: Settings, args: argparse.Namespace) -> None:
    kept = sorted((settings.cache_dir / "voice").glob("*-api.wav"))  # OMS_DEBUG_VOICE notes
    audio = args.audio or (kept[-1] if kept else None)
    if audio is None or not audio.is_file():
        raise SystemExit("pass --audio path/to/note (wav, webm, m4a…) or set OMS_DEBUG_VOICE=true")
    print(f"audio: {audio}")
    cache = DiskCache(settings.cache_dir)
    transcriber = Transcriber(make_client(settings), settings.transcribe_model, cache)
    text, call = await transcriber.transcribe(audio.read_bytes())
    print(f"model={settings.transcribe_model} cached={call.cached} latency_ms={call.latency_ms}")
    print(f"transcript: {text!r}")


Check = Callable[[Settings, argparse.Namespace], Coroutine[Any, Any, None]]
CHECKS: dict[str, Check] = {
    "spark": smoke_spark,
    "sam": smoke_sam,
    "vision": smoke_vision,
    "image": smoke_image,
    "open": smoke_open,
    "transcribe": smoke_transcribe,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("check", choices=sorted(CHECKS))
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="photo for vision checks")
    parser.add_argument("--phrase", default="box", help="noun phrase for the sam check")
    parser.add_argument(
        "--audio",
        type=Path,
        help="recording for the transcribe check (default: last kept web note)",
    )
    args = parser.parse_args(argv)
    asyncio.run(CHECKS[args.check](get_settings(), args))


if __name__ == "__main__":
    main()
