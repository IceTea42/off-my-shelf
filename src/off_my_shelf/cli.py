"""Command line: ``uv run off-my-shelf run photos/ --out packs/demo``."""

import argparse
import asyncio
from pathlib import Path

from off_my_shelf.config import get_settings
from off_my_shelf.export import export_pack
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.schema import Language


def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    services = build_services(settings, offline=args.offline, product_shots=not args.no_shots)
    pipeline = Pipeline(services, settings, progress=print)
    pack = asyncio.run(pipeline.run(args.inputs, args.out, Language(args.lang)))
    total = sum(call.cost_usd or 0.0 for item in pack.items for call in item.calls)
    print(f"{len(pack.items)} items -> {args.out / 'listing_pack.json'} (list cost ${total:.4f})")
    print(f"listing pack: {export_pack(args.out)}")


def _export(args: argparse.Namespace) -> None:
    print(f"listing pack: {export_pack(args.pack)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="off-my-shelf", description="Resale listings from media")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="analyse photos/videos into a listing pack")
    run.add_argument("inputs", nargs="+", type=Path, help="photo or video files, or folders")
    run.add_argument("--out", type=Path, required=True, help="output pack folder")
    run.add_argument("--lang", choices=[lang.value for lang in Language], default="en")
    run.add_argument("--offline", action="store_true", help="use cached results only")
    run.add_argument("--no-shots", action="store_true", help="skip AI product shots")
    run.set_defaults(handler=_run)

    export = commands.add_parser("export", help="rebuild index.html, listings and zip of a pack")
    export.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    export.set_defaults(handler=_export)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
