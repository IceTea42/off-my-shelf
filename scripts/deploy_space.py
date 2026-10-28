"""Publish the web app to a free Hugging Face Space (Gradio SDK, CPU basic).

A Gradio Space only runs ``app.py``; ours starts this project's own FastAPI app on the Space's
port, so no Gradio UI is involved. Only files tracked by git are uploaded (never ``.env``,
caches or packs), plus three generated files: the Space README header, ``app.py`` and
``requirements.txt`` (the locked runtime dependencies and the project).

Once:   uv run hf auth login                  (a token with "write" access)
Deploy: uv run python scripts/deploy_space.py <user>/off-my-shelf --set-secrets --budget 10
"""

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from off_my_shelf.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
GRADIO_VERSION = "6.31.0"  # required by the Space header; the app does not import Gradio
SPACE_HEADER = f"""---
title: Off My Shelf
emoji: 🏷️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.12"
app_file: app.py
license: agpl-3.0
short_description: Film your shelf, get a ready-to-post listing for every item.
---

"""
APP_PY = '''"""Hugging Face Space entry point: serve the Off My Shelf web app on port 7860."""

import os

os.environ.setdefault("OMS_HOST", "0.0.0.0")
os.environ.setdefault("OMS_PORT", "7860")
os.environ.setdefault("OMS_CACHE_DIR", "/tmp/off-my-shelf/cache")
os.environ.setdefault("OMS_RUNS_DIR", "/tmp/off-my-shelf/runs")

from off_my_shelf.cli import main

main(["serve"])
'''


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git arguments
        ["git", *args],  # noqa: S607 - git from PATH, as in any checkout
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def requirements() -> str:
    """The locked runtime dependencies as pip pins, then this project itself."""
    pins = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return pins + ".\n"


def space_operations() -> list[CommitOperationAdd]:
    """Tracked files plus the Space-only ones; README.md gets the Space header."""
    tracked = [name for name in _git("ls-files", "-z").split("\0") if name]
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    generated = {
        "README.md": readme,
        "app.py": APP_PY,
        "requirements.txt": requirements(),
    }
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=ROOT / name)
        for name in tracked
        if name not in generated
    ]
    operations += [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=text.encode("utf-8"))
        for name, text in generated.items()
    ]
    return operations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("space", help="<user>/<space name>, e.g. ogem/off-my-shelf")
    parser.add_argument("--set-secrets", action="store_true", help="copy META_API_KEY from .env")
    parser.add_argument("--budget", type=float, help="OMS_BUDGET_USD for the public app")
    args = parser.parse_args(argv)

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    if args.set_secrets:
        key = get_settings().meta_api_key
        if key is None:
            raise SystemExit("META_API_KEY is not set in .env")
        api.add_space_secret(args.space, "META_API_KEY", key.get_secret_value())
    if args.budget is not None:
        api.add_space_variable(args.space, "OMS_BUDGET_USD", f"{args.budget:g}")
    commit = api.create_commit(
        args.space,
        repo_type="space",
        operations=space_operations(),
        commit_message=f"Deploy {_git('rev-parse', '--short', 'HEAD').strip()}",
    )
    print(f"pushed {commit.oid[:8]}: https://huggingface.co/spaces/{args.space}")
    print("the Space builds in a few minutes; then open its App tab")


if __name__ == "__main__":
    main()
