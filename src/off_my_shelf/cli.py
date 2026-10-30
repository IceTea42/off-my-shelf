"""Command line: ``uv run off-my-shelf run photos/ --out packs/demo``."""

import argparse
import asyncio
from pathlib import Path

from off_my_shelf.agent import AgentRequest, run_agent
from off_my_shelf.config import get_settings
from off_my_shelf.demo import DEMO_PACK_DIR, make_demo
from off_my_shelf.evaluate import add_pack_to_eval, run_eval
from off_my_shelf.export import export_pack
from off_my_shelf.mocks import load_pack, save_pack
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.report import markdown_table
from off_my_shelf.schema import Language, TriageDecision
from off_my_shelf.tracing import setup_tracing
from off_my_shelf.video_ad import make_item_ad


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


def _ads(args: argparse.Namespace) -> None:
    settings = get_settings()
    spark = None if args.offline else build_services(settings).spark
    pack = load_pack(args.pack)
    for item in pack.items:
        if item.analysis.triage.decision is TriageDecision.SELL and item.ad_video is None:
            item.ad_video, call = asyncio.run(make_item_ad(spark, args.pack, item))
            item.calls = [*item.calls, *([call] if call else [])]
            print(f"{item.item_id}: {args.pack / item.ad_video}")
    save_pack(args.pack, pack)


def _make_demo(args: argparse.Namespace) -> None:
    files = make_demo(args.pack)
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"demo pack: {len(files)} files, {size:.1f} MB in {DEMO_PACK_DIR}")
    print("check every photo is fine to publish before committing it")


def _costs(args: argparse.Namespace) -> None:
    print(markdown_table(load_pack(args.pack)))


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


def _eval_add(args: argparse.Namespace) -> None:
    for hint in add_pack_to_eval(args.pack, args.dataset):
        print(hint)
    print(f"now fill name, brand and price_eur in {args.dataset} (blank rows are skipped)")


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

    ads = commands.add_parser("ads", help="10 s vertical video ad per item to sell")
    ads.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    ads.add_argument("--offline", action="store_true", help="no hook line (no API call)")
    ads.set_defaults(handler=_ads)

    demo = commands.add_parser("make-demo", help="bundle a real pack as the app's demo")
    demo.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    demo.set_defaults(handler=_make_demo)

    costs = commands.add_parser("costs", help="cost and latency per step of a pack (markdown)")
    costs.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    costs.set_defaults(handler=_costs)

    agent = commands.add_parser("agent", help="let the agent act on a pack from a goal")
    agent.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    agent.add_argument("goal", nargs="?", help='e.g. "moving Friday: sell fast, donate books"')
    agent.add_argument(
        "--voice", type=Path, help="voice note with the goal (wav, webm, ogg, m4a, mp3)"
    )
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

    eval_add = commands.add_parser("eval-add", help="add a real run's items to the eval set")
    eval_add.add_argument("pack", type=Path, help="pack folder containing listing_pack.json")
    eval_add.add_argument("--dataset", type=Path, default=Path("data/eval/items.csv"))
    eval_add.set_defaults(handler=_eval_add)

    serve = commands.add_parser("serve", help="start the web app")
    serve.add_argument("--host", help="interface, e.g. 0.0.0.0 to share on your network")
    serve.add_argument("--port", type=int)
    serve.set_defaults(handler=_serve)

    args = parser.parse_args(argv)
    setup_tracing()
    args.handler(args)


if __name__ == "__main__":
    main()
