"""Host the web app on Modal: an HTTPS address that also installs as the Android app.

Once:    uv run modal token new                     (opens your browser to log in)
Deploy:  uv run modal deploy scripts/modal_app.py   (prints the app's https://...modal.run URL)

The keys come from your local ``.env`` at deploy time (sent to Modal as an encrypted secret,
never committed). Runs and the model cache live on a Modal Volume, so they survive restarts.
One container serves everyone: runs are tracked in its memory, and it idles down after 20 min.
"""

from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parent.parent
DATA = PurePosixPath("/data")
BUDGET_USD = "10"  # live work stops once this much is spent; the demo pack keeps working

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0")  # OpenCV video decoding
    .uv_sync(str(ROOT), extra_options="--no-dev")  # the locked runtime dependencies
    # the package with its page, templates and fonts, importable from /root (on sys.path)
    .add_local_dir(ROOT / "src" / "off_my_shelf", "/root/off_my_shelf", ignore=["**/__pycache__"])
)
volume = modal.Volume.from_name("off-my-shelf-data", create_if_missing=True)
app = modal.App("off-my-shelf", image=image)


@app.function(
    secrets=[modal.Secret.from_dotenv(ROOT)],
    env={"OMS_BUDGET_USD": BUDGET_USD},
    volumes={DATA: volume},
    max_containers=1,
    scaledown_window=20 * 60,
    timeout=30 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web() -> object:
    from off_my_shelf.config import Settings  # noqa: PLC0415 - imported inside the container
    from off_my_shelf.web import create_app  # noqa: PLC0415

    return create_app(Settings(cache_dir=Path(DATA / "cache"), runs_dir=Path(DATA / "runs")))
