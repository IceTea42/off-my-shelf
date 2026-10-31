"""Web app: upload photos or videos, watch the run, browse the listings, download the pack.

One process, runs kept in memory and on disk under ``runs_dir``. Without META_API_KEY the app
runs offline: cached runs and the bundled demo pack still work.
"""

import asyncio
import logging
import mimetypes
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel, Field

from off_my_shelf.agent import (
    AgentAction,
    AgentRequest,
    SellerPolicy,
    apply_actions,
    run_agent,
)
from off_my_shelf.cache import DiskCache
from off_my_shelf.config import Settings, get_settings
from off_my_shelf.demo import demo_pack_dir
from off_my_shelf.export import export_pack
from off_my_shelf.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from off_my_shelf.meta_client import describe_error, make_client
from off_my_shelf.mocks import PACK_FILENAME, load_pack, save_pack
from off_my_shelf.pipeline import Pipeline, build_services
from off_my_shelf.report import live_cost
from off_my_shelf.schema import Item, Language, ListingPack
from off_my_shelf.tracing import trace_app
from off_my_shelf.trials import (
    RUNS_PER_HOUR,
    VISITOR_ID,
    RateLimiter,
    Trial,
    load_trial,
    prune,
    save_trial,
    visitor_trials,
)
from off_my_shelf.video_ad import make_item_ad
from off_my_shelf.voice import AudioFormatError, Transcriber, to_api_wav, wav_seconds

STATIC_DIR = Path(__file__).parent / "static"
DEMO_RUN_ID = "demo"
MAX_UPLOAD_FILES = 20
MIN_VOICE_SECONDS = 0.5
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
VISITOR_COOKIE = "oms_visitor"
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


class AgentForm(BaseModel):
    """The agent panel: a goal and/or a buyer message, and the seller's own rules."""

    goal: str = ""
    buyer_item_id: str = ""
    buyer_message: str = ""
    never_sell: str = ""  # comma- or line-separated item ids or words of titles
    max_discount_pct: float | None = Field(default=None, ge=0, le=90)

    def policy(self) -> SellerPolicy:
        terms = [term.strip() for term in re.split(r"[,\n]", self.never_sell) if term.strip()]
        discount = None if self.max_discount_pct is None else self.max_discount_pct / 100
        return SellerPolicy(never_sell=terms, max_discount=discount)


class Proposal(BaseModel):
    """What the agent proposed in one session, kept until a person applies or replaces it."""

    policy: SellerPolicy
    actions: list[AgentAction]


PROPOSAL_FILE = "proposal.json"


def _agent_dir(pack_dir: Path) -> Path:
    """Agent proposals and undo snapshots, next to the pack (not inside it, nor in its zip)."""
    return pack_dir.parent / "agent"


def visitor_of(request: Request) -> str:
    """The anonymous id the page keeps in this browser (cookie), or one for this address."""
    cookie = request.cookies.get(VISITOR_COOKIE, "")
    if VISITOR_ID.match(cookie):
        return cookie
    return f"ip-{request.client.host if request.client else 'unknown'}"


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


def _suffix(upload: UploadFile) -> str:
    """The file's extension, or one guessed from its type (phone galleries may share names
    without one, e.g. ``IMG_0042``)."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in MEDIA_SUFFIXES or not upload.content_type:
        return suffix
    return (mimetypes.guess_extension(upload.content_type) or suffix).lower()


def _save_uploads(files: list[UploadFile], upload_dir: Path) -> None:
    """Store uploads under neutral names; reject the whole batch on any unsupported file."""
    suffixes = [_suffix(upload) for upload in files]
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
    limiter: RateLimiter = field(default_factory=RateLimiter)  # per client address

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

    def start_run(
        self, files: list[UploadFile], language: Language, product_shots: bool, request: Request
    ) -> str:
        """Save the uploads and run the pipeline in the background; returns the run id."""
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(400, f"send 1 to {MAX_UPLOAD_FILES} photos or videos")
        if not self.offline:  # offline runs replay the cache for free
            self.require_live()
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                raise HTTPException(429, f"Up to {RUNS_PER_HOUR} runs an hour: try again later.")
        run_id = uuid.uuid4().hex[:12]
        prune(self.settings.runs_dir)
        save_trial(
            self.settings.runs_dir,
            Trial(run_id=run_id, visitor=visitor_of(request), created_at=datetime.now(UTC)),
        )
        upload_dir = self.settings.runs_dir / run_id / "uploads"
        _save_uploads(files, upload_dir)
        run = self.runs[run_id] = Run(run_id)
        services = build_services(self.settings, offline=self.offline, product_shots=product_shots)
        pipeline = Pipeline(services, self.settings, progress=run.messages.append)
        run.pipeline = pipeline

        async def work() -> None:
            try:
                pack = await pipeline.run([upload_dir], self.pack_dir(run_id), language)
                self.spent_usd += live_cost(pack)
                await asyncio.to_thread(export_pack, self.pack_dir(run_id))
                run.status = RunStatus.DONE
            except Exception as error:
                run.status, run.error = RunStatus.ERROR, f"{type(error).__name__}: {error}"

        task = asyncio.create_task(work())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return run_id

    def item(self, run_id: str, item_id: str) -> tuple[Path, ListingPack, Item]:
        """A finished run's pack folder, its pack and one of its items (404 otherwise)."""
        target = self.finished(run_id)
        pack = load_pack(target)
        found = next((i for i in pack.items if i.item_id == item_id), None)
        if found is None:
            raise HTTPException(404, "unknown item")
        return target, pack, found

    def lookup(self, run_id: str) -> Run | None:
        """A run in memory, or a finished one found on disk (after a restart)."""
        on_disk = (self.pack_dir(run_id) / PACK_FILENAME).is_file()
        if run_id not in self.runs and on_disk and load_trial(self.settings.runs_dir, run_id):
            self.runs[run_id] = Run(run_id, RunStatus.DONE, ["run reopened"])
        return self.runs.get(run_id)

    def finished(self, run_id: str) -> Path:
        run = self.lookup(run_id)
        if run is None or run.status is not RunStatus.DONE:
            raise HTTPException(404, "pack not ready")
        return self.pack_dir(run_id)


def create_app(settings: Settings | None = None) -> FastAPI:
    state = _State(settings or get_settings(), limiter=RateLimiter())
    state.settings.runs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Off My Shelf")
    trace_app(app, state.settings.logfire_token)
    app.mount("/runs", StaticFiles(directory=state.settings.runs_dir), name="runs")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    _page_routes(app, state)
    _run_routes(app, state)
    _voice_routes(app, state)
    _item_routes(app, state)
    _agent_routes(app, state)
    _trial_routes(app, state)
    _proposal_routes(app, state)
    return app


def _page_routes(app: FastAPI, state: _State) -> None:
    @app.get("/favicon.svg")
    async def favicon_svg() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/favicon.ico")
    async def favicon_ico() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")

    @app.get("/manifest.webmanifest")
    async def manifest() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json"
        )

    @app.get("/sw.js")
    async def service_worker() -> FileResponse:
        """Served from the root so the installed app's scope is the whole site."""
        return FileResponse(
            STATIC_DIR / "sw.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

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
        request: Request,
        files: Annotated[list[UploadFile], File()],
        language: Annotated[Language, Form()] = Language.EN,
        product_shots: Annotated[bool, Form()] = True,
    ) -> dict[str, str]:
        return {"run_id": state.start_run(files, language, product_shots, request)}

    @app.post("/share")
    async def share_target(
        request: Request, files: Annotated[list[UploadFile], File()]
    ) -> RedirectResponse:
        """Android share sheet ("Share" photos to the installed app): start a run, show it."""
        language = state.settings.default_language
        run_id = state.start_run(files, language, product_shots=True, request=request)
        return RedirectResponse(f"/?run={run_id}", status_code=303)

    @app.post("/api/demo")
    async def demo() -> dict[str, str]:
        """The bundled mock pack as a finished run, for a demo without any API call."""
        target = state.pack_dir(DEMO_RUN_ID)
        if not target.exists():
            shutil.copytree(demo_pack_dir(), target)
            await asyncio.to_thread(export_pack, target)
        state.runs[DEMO_RUN_ID] = Run(DEMO_RUN_ID, RunStatus.DONE, ["demo pack loaded"])
        return {"run_id": DEMO_RUN_ID}

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str, since: int = 0) -> dict[str, Any]:
        """Run state; ``messages`` holds only those after the first ``since``."""
        run = state.lookup(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        pack = None
        if run.status is RunStatus.DONE:
            pack = load_pack(state.pack_dir(run_id)).model_dump(mode="json")
        partial = run.pipeline.partial if run.pipeline and run.status is RunStatus.RUNNING else {}
        return {
            "run_id": run_id,
            "status": run.status.value,
            "messages": run.messages[max(since, 0) :],
            "message_count": len(run.messages),
            "error": run.error,
            "items_done": run.pipeline.items_done if run.pipeline else 0,
            "items_total": run.pipeline.items_total if run.pipeline else 0,
            "pack": pack,
            "items": [partial[key].model_dump(mode="json") for key in sorted(partial)],
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


def _item_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/runs/{run_id}/items/{item_id}/translate")
    async def translate(
        run_id: str, item_id: str, language: Annotated[Language, Form()]
    ) -> dict[str, Any]:
        """The item's listing in another language, made once and kept in the pack."""
        target, pack, item = state.item(run_id, item_id)
        listing = item.analysis.listing
        known = [listing, *item.translations]
        done = next((text for text in known if text.language is language), None)
        if done is not None:
            return done.model_dump(mode="json")
        state.require_live()
        spark = build_services(state.settings, product_shots=False).spark
        try:
            result = await spark.translate(listing, language)
        except OpenAIError as error:
            raise HTTPException(502, f"Translation failed: {describe_error(error)}") from error
        item.translations = [*item.translations, result.value]
        item.calls = [*item.calls, result.call]
        save_pack(target, pack)
        await asyncio.to_thread(export_pack, target)
        state.spent_usd += 0.0 if result.call.cached else result.call.cost_usd or 0.0
        return result.value.model_dump(mode="json")

    @app.post("/api/runs/{run_id}/items/{item_id}/ad")
    async def video_ad(run_id: str, item_id: str) -> dict[str, str]:
        """A 10 s vertical MP4 ad from the item's real shot; offline it has no hook line."""
        target, pack, item = state.item(run_id, item_id)
        if item.ad_video is None:
            if not state.offline:
                state.require_live()
            spark = None if state.offline else build_services(state.settings).spark
            try:
                item.ad_video, call = await make_item_ad(spark, target, item)
            except OpenAIError as error:
                raise HTTPException(502, f"Video ad failed: {describe_error(error)}") from error
            if call is not None:
                item.calls = [*item.calls, call]
                state.spent_usd += 0.0 if call.cached else call.cost_usd or 0.0
            save_pack(target, pack)
        return {"video": item.ad_video}


def _proposal_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/runs/{run_id}/agent/apply")
    async def apply_proposal(run_id: str, selected: Annotated[str, Form()] = "") -> dict[str, Any]:
        """Apply the actions a person picked (indexes into the proposal), checked again."""
        target = state.finished(run_id)
        proposal_file = _agent_dir(target) / PROPOSAL_FILE
        if not proposal_file.is_file():
            raise HTTPException(404, "no proposal to apply")
        proposal = Proposal.model_validate_json(proposal_file.read_text(encoding="utf-8"))
        picks = {int(i) for i in selected.split(",") if i.strip().isdigit()}
        chosen = [a for n, a in enumerate(proposal.actions) if n in picks]
        pack = load_pack(target)
        history = _agent_dir(target) / "history"
        history.mkdir(exist_ok=True)
        (history / f"{len(list(history.glob('*.json'))):03d}.json").write_text(
            pack.model_dump_json(), encoding="utf-8"
        )
        applied, log = apply_actions(pack, chosen, proposal.policy)
        save_pack(target, applied)
        proposal_file.unlink()
        await asyncio.to_thread(export_pack, target)
        return {"actions": log, "pack": applied.model_dump(mode="json"), "can_undo": True}

    @app.post("/api/runs/{run_id}/agent/undo")
    async def undo(run_id: str) -> dict[str, Any]:
        """Put the pack back as it was before the last applied agent session."""
        target = state.finished(run_id)
        snapshots = sorted((_agent_dir(target) / "history").glob("*.json"))
        if not snapshots:
            raise HTTPException(404, "nothing to undo")
        pack = ListingPack.model_validate_json(snapshots[-1].read_text(encoding="utf-8"))
        save_pack(target, pack)
        snapshots[-1].unlink()
        await asyncio.to_thread(export_pack, target)
        return {"pack": pack.model_dump(mode="json"), "can_undo": len(snapshots) > 1}


def _trial_routes(app: FastAPI, state: _State) -> None:
    @app.get("/api/trials")
    async def my_trials(request: Request) -> list[dict[str, object]]:
        """This browser's own runs, newest first (hidden ones left out)."""
        return visitor_trials(state.settings.runs_dir, visitor_of(request))

    @app.post("/api/trials/{run_id}/hide")
    async def hide_trial(run_id: str, request: Request) -> dict[str, bool]:
        """Hide a run from its owner's list; it stays in the rolling buffer until it expires."""
        trial = load_trial(state.settings.runs_dir, run_id)
        if trial is None or trial.visitor != visitor_of(request):
            raise HTTPException(404, "unknown run")
        save_trial(state.settings.runs_dir, trial.model_copy(update={"hidden": True}))
        return {"hidden": True}


def _agent_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/runs/{run_id}/agent")
    async def agent(run_id: str, form: Annotated[AgentForm, Form()]) -> dict[str, Any]:
        """The agent proposes checked actions; nothing changes until a person applies them."""
        target = state.finished(run_id)
        state.require_live()
        policy = form.policy()
        request = AgentRequest(
            goal=form.goal,
            buyer_messages=(
                {form.buyer_item_id: form.buyer_message}
                if form.buyer_item_id and form.buyer_message
                else {}
            ),
            policy=policy,
        )
        try:
            outcome = await run_agent(state.settings, target, request, apply=False)
        except Exception as error:
            detail = describe_error(error)
            logger.exception("agent failed: %s", detail)
            raise HTTPException(502, f"Agent failed: {detail}") from error
        state.spent_usd += outcome.cost_usd
        _agent_dir(target).mkdir(exist_ok=True)
        (_agent_dir(target) / PROPOSAL_FILE).write_text(
            Proposal(policy=policy, actions=outcome.proposal).model_dump_json(),
            encoding="utf-8",
        )
        return {
            "goal": outcome.goal,
            "actions": outcome.actions,
            "summary": outcome.summary,
            "proposal": [action.text for action in outcome.proposal],
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
