"""Command line: ``uv run off-my-shelf run photos/ --out packs/demo``."""

import argparse
import asyncio
from pathlib import Path

from off_my_shelf.agent import AgentRequest, run_agent
from off_my_shelf.config import get_settings
from off_my_shelf.evaluate import run_eval
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


def _agent(args: argparse.Namespace) -> None:
    request = AgentRequest(
        goal=args.goal or "",
        voice=args.voice.read_bytes() if args.voice else None,
        buyer_messages=dict(args.buyer or []),
    )
    outcome = asyncio.run(run_agent(get_settings(), args.pack, request, offline=args.offline))
    print(f"goal: {outcome.goal}")
    for action in outcome.actions:
        print(f"  - {action}")
    print(outcome.summary)


def _eval(args: argparse.Namespace) -> None:
    settings = get_settings()
    services = build_services(settings, offline=args.offline, product_shots=False)
    summary = asyncio.run(run_eval(services, settings, args.dataset, args.out))
    for metric, value in summary.items():
        print(f"{metric}: {value}")
    print(f"report: {args.out / 'report.md'}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn  # noqa: PLC0415 - only the serve command needs the web stack

    from off_my_shelf.web import create_app  # noqa: PLC0415

    settings = get_settings()
    host, port = args.host or settings.host, args.port or settings.port
    print(f"Off My Shelf on http://{host}:{port}")
    uvicorn.run(create_app(settings), host=host, port=port)


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

    agent = commands.add_parser("agent", help="let the agent act on a pack from a goal")
    agent.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    agent.add_argument("goal", nargs="?", help='e.g. "moving Friday: sell fast, donate books"')
    agent.add_argument("--voice", type=Path, help="WAV voice note with the goal")
    agent.add_argument(
        "--buyer",
        nargs=2,
        action="append",
        metavar=("ITEM_ID", "MESSAGE"),
        help="a buyer's message to answer (repeatable)",
    )
    agent.add_argument("--offline", action="store_true", help="replay cached steps only")
    agent.set_defaults(handler=_agent)

    evaluation = commands.add_parser("eval", help="score the pipeline on known resale prices")
    evaluation.add_argument("dataset", type=Path, help="CSV: image,name,brand,price_eur")
    evaluation.add_argument("--out", type=Path, default=Path("reports/eval"))
    evaluation.add_argument("--offline", action="store_true", help="use cached results only")
    evaluation.set_defaults(handler=_eval)

    serve = commands.add_parser("serve", help="start the web app")
    serve.add_argument("--host", help="interface, e.g. 0.0.0.0 to share on your network")
    serve.add_argument("--port", type=int)
    serve.set_defaults(handler=_serve)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
