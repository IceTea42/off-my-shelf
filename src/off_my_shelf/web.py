"""Web app: upload photos or videos, watch the run, browse the listings, download the pack.

One process, runs kept in memory and on disk under ``runs_dir``. Without META_API_KEY the app
runs offline: cached runs and the bundled demo pack still work.
"""

import asyncio
import logging
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from off_my_shelf.agent import AgentRequest, run_agent
from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings, get_settings
from off_my_shelf.export import export_pack
from off_my_shelf.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from off_my_shelf.meta_client import describe_error, make_client
from off_my_shelf.mocks import MOCK_PACK_DIR, load_pack, save_pack
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.report import live_cost
from off_my_shelf.schema import Language
from off_my_shelf.voice import AudioFormatError, Transcriber, to_api_wav, wav_seconds

STATIC_DIR = Path(__file__).parent / "static"
DEMO_RUN_ID = "demo"
MAX_UPLOAD_FILES = 20
MIN_VOICE_SECONDS = 0.5
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
OFFLINE_DETAIL = "This needs META_API_KEY in .env (the app is in offline mode)."
BUDGET_DETAIL = "This demo's API budget is used up: try the demo pack, or run it locally."
logger = logging.getLogger(__name__)


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
    pipeline: Pipeline | None = None


def _voice_keeper(settings: Settings, suffix: str) -> Callable[[str, bytes], None]:
    """Debug only: a saver for one voice note's files, or a no-op when debugging is off."""
    if not settings.debug_voice:
        return lambda _name, _data: None
    folder = settings.cache_dir / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    def keep(name: str, data: bytes) -> None:
        filename = f"{stamp}-upload{suffix.lower()}" if name == "upload" else f"{stamp}-{name}"
        (folder / filename).write_bytes(data)

    return keep


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


@dataclass
class _State:
    """What the routes share: settings, the runs in memory and their background tasks."""

    settings: Settings
    runs: dict[str, Run] = field(default_factory=dict)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    spent_usd: float = 0.0

    @property
    def offline(self) -> bool:
        return self.settings.meta_api_key is None

    def require_live(self) -> None:
        """Live calls need a key and, on a public deployment, budget left."""
        if self.offline:
            raise HTTPException(503, OFFLINE_DETAIL)
        budget = self.settings.budget_usd
        if budget is not None and self.spent_usd >= budget:
            raise HTTPException(429, BUDGET_DETAIL)

    def pack_dir(self, run_id: str) -> Path:
        return self.settings.runs_dir / run_id / "pack"

    def finished(self, run_id: str) -> Path:
        run = self.runs.get(run_id)
        if run is None or run.status is not RunStatus.DONE:
            raise HTTPException(404, "pack not ready")
        return self.pack_dir(run_id)


def create_app(settings: Settings | None = None) -> FastAPI:
    state = _State(settings or get_settings())
    state.settings.runs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Off My Shelf")
    app.mount("/runs", StaticFiles(directory=state.settings.runs_dir), name="runs")
    _page_routes(app, state)
    _run_routes(app, state)
    _voice_routes(app, state)
    _agent_routes(app, state)
    return app


def _page_routes(app: FastAPI, state: _State) -> None:
    @app.get("/favicon.svg")
    async def favicon_svg() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/favicon.ico")
    async def favicon_ico() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "offline": state.offline,
            "languages": [lang.value for lang in Language],
            "default_language": state.settings.default_language.value,
        }


def _run_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/runs", status_code=202)
    async def start_run(
        files: Annotated[list[UploadFile], File()],
        language: Annotated[Language, Form()] = Language.EN,
        product_shots: Annotated[bool, Form()] = True,
    ) -> dict[str, str]:
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(400, f"send 1 to {MAX_UPLOAD_FILES} photos or videos")
        if not state.offline:  # offline runs replay the cache for free
            state.require_live()
        run_id = uuid.uuid4().hex[:12]
        upload_dir = state.settings.runs_dir / run_id / "uploads"
        _save_uploads(files, upload_dir)
        run = state.runs[run_id] = Run(run_id)
        services = build_services(
            state.settings, offline=state.offline, product_shots=product_shots
        )
        pipeline = Pipeline(services, state.settings, progress=run.messages.append)
        run.pipeline = pipeline

        async def work() -> None:
            try:
                pack = await pipeline.run([upload_dir], state.pack_dir(run_id), language)
                state.spent_usd += live_cost(pack)
                await asyncio.to_thread(export_pack, state.pack_dir(run_id))
                run.status = RunStatus.DONE
            except Exception as error:
                run.status, run.error = RunStatus.ERROR, f"{type(error).__name__}: {error}"

        task = asyncio.create_task(work())
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        return {"run_id": run_id}

    @app.post("/api/demo")
    async def demo() -> dict[str, str]:
        """The bundled mock pack as a finished run, for a demo without any API call."""
        target = state.pack_dir(DEMO_RUN_ID)
        if not target.exists():
            shutil.copytree(MOCK_PACK_DIR, target)
            await asyncio.to_thread(export_pack, target)
        state.runs[DEMO_RUN_ID] = Run(DEMO_RUN_ID, RunStatus.DONE, ["demo pack loaded"])
        return {"run_id": DEMO_RUN_ID}

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> dict[str, Any]:
        run = state.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        pack = None
        if run.status is RunStatus.DONE:
            pack = load_pack(state.pack_dir(run_id)).model_dump(mode="json")
        return {
            "run_id": run_id,
            "status": run.status.value,
            "messages": run.messages[-20:],
            "error": run.error,
            "items_done": run.pipeline.items_done if run.pipeline else 0,
            "items_total": run.pipeline.items_total if run.pipeline else 0,
            "warnings": run.pipeline.warnings if run.pipeline else [],
            "pack": pack,
            "base_url": f"/runs/{run_id}/pack/",
        }

    @app.get("/api/runs/{run_id}/zip")
    async def download(run_id: str) -> FileResponse:
        archive = state.finished(run_id).with_suffix(".zip")
        if not archive.is_file():
            raise HTTPException(404, "pack not ready")
        return FileResponse(archive, filename=f"off-my-shelf-{run_id}.zip")


def _voice_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/transcribe")
    async def transcribe(voice: Annotated[UploadFile, File()]) -> dict[str, Any]:
        """A spoken goal as text, shown to the person before the agent acts on it.

        With ``OMS_DEBUG_VOICE=true`` every note is kept under ``<cache>/voice`` (the upload,
        the WAV sent to the API and the outcome); replay one with ``smoke transcribe``.
        """
        state.require_live()
        settings = state.settings
        upload = await voice.read()
        keep = _voice_keeper(settings, Path(voice.filename or "note.webm").suffix)
        keep("upload", upload)
        try:
            audio = await asyncio.to_thread(to_api_wav, upload)
        except AudioFormatError as error:
            keep("result.txt", f"unreadable: {error}".encode())
            raise HTTPException(
                400, f"Recording unreadable ({voice.content_type}): {error}"
            ) from error
        keep("api.wav", audio)
        seconds = wav_seconds(audio)
        if seconds < MIN_VOICE_SECONDS:
            raise HTTPException(
                400, f"Recording too short ({seconds:.1f} s): hold on a bit longer."
            )
        transcriber = Transcriber(
            make_client(settings), settings.transcribe_model, DiskCache(settings.cache_dir)
        )
        try:
            text, call = await transcriber.transcribe(audio)
        except OpenAIError as error:
            detail = describe_error(error)
            keep("result.txt", detail.encode())
            logger.exception("Voice Transcribe failed: %s", detail)
            raise HTTPException(502, f"Voice Transcribe failed: {detail}") from error
        keep("result.txt", text.encode())
        state.spent_usd += 0.0 if call.cached else call.cost_usd or 0.0
        if not text:
            raise HTTPException(
                422, f"No speech recognised in {seconds:.1f} s of audio: check the microphone."
            )
        return {"transcript": text, "seconds": round(seconds, 1)}


def _agent_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/runs/{run_id}/agent")
    async def agent(
        run_id: str,
        goal: Annotated[str, Form()] = "",
        buyer_item_id: Annotated[str, Form()] = "",
        buyer_message: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        """Let the agent act on a finished run from a goal and/or a buyer message."""
        target = state.finished(run_id)
        state.require_live()
        before = live_cost(load_pack(target))
        request = AgentRequest(
            goal=goal,
            buyer_messages=(
                {buyer_item_id: buyer_message} if buyer_item_id and buyer_message else {}
            ),
        )
        try:
            outcome = await run_agent(state.settings, target, request)
        except Exception as error:
            detail = describe_error(error)
            logger.exception("agent failed: %s", detail)
            raise HTTPException(502, f"Agent failed: {detail}") from error
        pack = load_pack(target)
        state.spent_usd += max(live_cost(pack) - before, 0.0)
        return {
            "goal": outcome.goal,
            "actions": outcome.actions,
            "summary": outcome.summary,
            "pack": pack.model_dump(mode="json"),
        }

    @app.post("/api/runs/{run_id}/items/{item_id}/replies/{index}/approve")
    async def approve_reply(run_id: str, item_id: str, index: int) -> dict[str, bool]:
        """A person approves a drafted buyer reply; only then may it be sent."""
        target = state.finished(run_id)
        pack = load_pack(target)
        for item in pack.items:
            if item.item_id == item_id and 0 <= index < len(item.buyer_replies):
                item.buyer_replies[index].approved = True
                save_pack(target, pack)
                await asyncio.to_thread(export_pack, target)
                return {"approved": True}
        raise HTTPException(404, "unknown reply")
