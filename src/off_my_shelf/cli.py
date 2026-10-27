"""Command line: ``uv run off-my-shelf run photos/ --out packs/demo``."""

import argparse
import asyncio
from pathlib import Path

from off_my_shelf.config import get_settings
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.schema import Language


def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    services = build_services(settings, offline=args.offline)
    pipeline = Pipeline(services, settings, progress=print)
    pack = asyncio.run(pipeline.run(args.inputs, args.out, Language(args.lang)))
    total = sum(call.cost_usd or 0.0 for item in pack.items for call in item.calls)
    print(f"{len(pack.items)} items -> {args.out / 'listing_pack.json'} (list cost ${total:.4f})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="off-my-shelf", description="Resale listings from media")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="analyse photos/videos into a listing pack")
    run.add_argument("inputs", nargs="+", type=Path, help="photo or video files, or folders")
    run.add_argument("--out", type=Path, required=True, help="output pack folder")
    run.add_argument("--lang", choices=[lang.value for lang in Language], default="en")
    run.add_argument("--offline", action="store_true", help="use cached results only")
    run.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
