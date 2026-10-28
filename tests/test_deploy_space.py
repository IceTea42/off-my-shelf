"""Space deploy script: what gets uploaded (no network: nothing is pushed)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "deploy_space.py"


def test_space_files_are_tracked_files_plus_space_entry_points() -> None:
    spec = importlib.util.spec_from_file_location("deploy_space", SCRIPT)
    assert spec is not None and spec.loader is not None
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)

    files = {op.path_in_repo: op.path_or_fileobj for op in deploy.space_operations()}

    assert ".env" not in files and not any(name.startswith(".cache") for name in files)
    assert "src/off_my_shelf/web.py" in files
    readme = files["README.md"]
    assert isinstance(readme, bytes) and readme.startswith(b"---\ntitle: Off My Shelf")
    assert b"app_file: app.py" in readme
    requirements = files["requirements.txt"]
    assert isinstance(requirements, bytes) and b"fastapi>=" in requirements
    assert b"==" not in requirements  # ranges, so pip can satisfy the Space's own Gradio pins
    assert not any(line.startswith(b".") for line in requirements.splitlines())
    assert not any(name.startswith(deploy.PRIVATE_DIRS) for name in files)
    assert b'main(["serve"])' in files["app.py"]
