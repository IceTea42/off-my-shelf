"""Space deploy script: what gets uploaded (no network: nothing is pushed)."""

import importlib.util
from pathlib import Path

from pydantic import SecretStr

from off_my_shelf.config import Settings

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


def test_modal_app_defines_the_web_endpoint() -> None:
    spec = importlib.util.spec_from_file_location("modal_app", SCRIPT.parent / "modal_app.py")
    assert spec is not None and spec.loader is not None
    modal_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modal_app)
    assert modal_app.app.name == "off-my-shelf"
    assert modal_app.BUDGET_USD == "10"


def test_space_secrets_carry_partner_keys_when_set() -> None:
    spec = importlib.util.spec_from_file_location("deploy_space", SCRIPT)
    assert spec is not None and spec.loader is not None
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        meta_api_key=SecretStr("m"),
        baseten_api_key=SecretStr("b"),
        logfire_token=SecretStr("l"),
    )
    assert deploy.space_secrets(settings) == {
        "META_API_KEY": "m",
        "BASETEN_API_KEY": "b",
        "LOGFIRE_TOKEN": "l",
    }
