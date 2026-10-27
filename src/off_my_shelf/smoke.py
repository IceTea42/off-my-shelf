"""One cheap live call per model, to check the key, the endpoint and the response shape.

Usage:
    uv run python -m off_my_shelf.smoke spark
    uv run python -m off_my_shelf.smoke sam --image path/to/photo.jpg --phrase mug
"""

import argparse
import asyncio
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings, get_settings
from off_my_shelf.costs import token_cost
from off_my_shelf.imaging import load_image
from off_my_shelf.meta_client import make_client
from off_my_shelf.mocks import MOCK_PACK_DIR
from off_my_shelf.sam import SamSegmenter

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


Check = Callable[[Settings, argparse.Namespace], Coroutine[Any, Any, None]]
CHECKS: dict[str, Check] = {"spark": smoke_spark, "sam": smoke_sam}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("check", choices=sorted(CHECKS))
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="photo for vision checks")
    parser.add_argument("--phrase", default="box", help="noun phrase for the sam check")
    args = parser.parse_args(argv)
    asyncio.run(CHECKS[args.check](get_settings(), args))


if __name__ == "__main__":
    main()
