"""Web app: upload photos or videos, watch the run, browse the listings, download the pack.

One process, runs kept in memory and on disk under ``runs_dir``. Without META_API_KEY the app
runs offline: cached runs and the bundled demo pack still work.
"""

import asyncio
import shutil
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from off_my_shelf.config import Settings, get_settings
from off_my_shelf.export import export_pack
from off_my_shelf.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from off_my_shelf.mocks import MOCK_PACK_DIR, load_pack
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.schema import Language

STATIC_DIR = Path(__file__).parent / "static"
DEMO_RUN_ID = "demo"
MAX_UPLOAD_FILES = 20
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


class RunStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Run:
    run_id: str
    status: RunStatus = RunStatus.RUNNING
    messages: list[str] = field(default_factory=list)
    error: str | None = None


def _save_uploads(files: list[UploadFile], upload_dir: Path) -> None:
    """Store uploads under neutral names; reject the whole batch on any unsupported file."""
    suffixes = [Path(upload.filename or "").suffix.lower() for upload in files]
    bad = [u.filename for u, s in zip(files, suffixes, strict=True) if s not in MEDIA_SUFFIXES]
    if bad:
        raise HTTPException(400, f"unsupported file: {', '.join(map(str, bad))}")
    upload_dir.mkdir(parents=True)
    for number, (upload, suffix) in enumerate(zip(files, suffixes, strict=True), start=1):
        with (upload_dir / f"upload-{number:02d}{suffix}").open("wb") as out:
            shutil.copyfileobj(upload.file, out)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    runs_dir = settings.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Run] = {}
    tasks: set[asyncio.Task[None]] = set()
    offline = settings.meta_api_key is None

    app = FastAPI(title="Off My Shelf")
    app.mount("/runs", StaticFiles(directory=runs_dir), name="runs")

    def pack_dir(run_id: str) -> Path:
        return runs_dir / run_id / "pack"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "offline": offline,
            "languages": [lang.value for lang in Language],
            "default_language": settings.default_language.value,
        }

    @app.post("/api/runs", status_code=202)
    async def start_run(
        files: Annotated[list[UploadFile], File()],
        language: Annotated[Language, Form()] = Language.EN,
        product_shots: Annotated[bool, Form()] = True,
    ) -> dict[str, str]:
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(400, f"send 1 to {MAX_UPLOAD_FILES} photos or videos")
        run_id = uuid.uuid4().hex[:12]
        upload_dir = runs_dir / run_id / "uploads"
        _save_uploads(files, upload_dir)
        run = runs[run_id] = Run(run_id)
        services = build_services(settings, offline=offline, product_shots=product_shots)
        pipeline = Pipeline(services, settings, progress=run.messages.append)

        async def work() -> None:
            try:
                await pipeline.run([upload_dir], pack_dir(run_id), language)
                await asyncio.to_thread(export_pack, pack_dir(run_id))
                run.status = RunStatus.DONE
            except Exception as error:
                run.status, run.error = RunStatus.ERROR, f"{type(error).__name__}: {error}"

        task = asyncio.create_task(work())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"run_id": run_id}

    @app.post("/api/demo")
    async def demo() -> dict[str, str]:
        """The bundled mock pack as a finished run, for a demo without any API call."""
        target = pack_dir(DEMO_RUN_ID)
        if not target.exists():
            shutil.copytree(MOCK_PACK_DIR, target)
            await asyncio.to_thread(export_pack, target)
        runs[DEMO_RUN_ID] = Run(DEMO_RUN_ID, RunStatus.DONE, ["demo pack loaded"])
        return {"run_id": DEMO_RUN_ID}

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> dict[str, Any]:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        pack = None
        if run.status is RunStatus.DONE:
            pack = load_pack(pack_dir(run_id)).model_dump(mode="json")
        return {
            "run_id": run_id,
            "status": run.status.value,
            "messages": run.messages[-20:],
            "error": run.error,
            "pack": pack,
            "base_url": f"/runs/{run_id}/pack/",
        }

    @app.get("/api/runs/{run_id}/zip")
    async def download(run_id: str) -> FileResponse:
        run = runs.get(run_id)
        archive = pack_dir(run_id).with_suffix(".zip")
        if run is None or run.status is not RunStatus.DONE or not archive.is_file():
            raise HTTPException(404, "pack not ready")
        return FileResponse(archive, filename=f"off-my-shelf-{run_id}.zip")

    return app
