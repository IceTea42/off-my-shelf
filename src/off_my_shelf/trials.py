"""My trials: who started which run, kept as a rolling buffer, hidden on request.

Each run folder gets a ``trial.json`` naming the browser that started it (an anonymous id kept in
that browser). People see and reopen only their own runs and can hide them; hidden runs stay in
the buffer for debugging until it rolls over: the newest ``MAX_RUNS`` runs, none older than
``MAX_AGE``. A per-visitor rate limit keeps a public deployment's budget for many people.
"""

import re
import shutil
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import AwareDatetime, BaseModel

from off_my_shelf.mocks import PACK_FILENAME, load_pack

TRIAL_FILE = "trial.json"
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
VISITOR_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MAX_RUNS = 300
MAX_AGE = timedelta(days=30)
RUNS_PER_HOUR = 10


class Trial(BaseModel):
    run_id: str
    visitor: str
    created_at: AwareDatetime
    hidden: bool = False


def pack_dir(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id / "pack"


def save_trial(runs_dir: Path, trial: Trial) -> None:
    path = runs_dir / trial.run_id / TRIAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trial.model_dump_json(), encoding="utf-8")


def load_trial(runs_dir: Path, run_id: str) -> Trial | None:
    if not RUN_ID.match(run_id):
        return None
    path = runs_dir / run_id / TRIAL_FILE
    return Trial.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def all_trials(runs_dir: Path) -> list[Trial]:
    """Every recorded run, newest first."""
    trials = [load_trial(runs_dir, folder.name) for folder in runs_dir.glob("*") if folder.is_dir()]
    return sorted((t for t in trials if t), key=lambda t: t.created_at, reverse=True)


def visitor_trials(runs_dir: Path, visitor: str) -> list[dict[str, object]]:
    """The visitor's visible runs, newest first, with what a list row needs."""
    rows: list[dict[str, object]] = []
    for trial in all_trials(runs_dir):
        if trial.visitor != visitor or trial.hidden:
            continue
        done = (pack_dir(runs_dir, trial.run_id) / PACK_FILENAME).is_file()
        pack = load_pack(pack_dir(runs_dir, trial.run_id)) if done else None
        rows.append(
            {
                "run_id": trial.run_id,
                "created_at": trial.created_at.isoformat(),
                "items": len(pack.items) if pack else None,
                "thumbnail": pack.media[0] if pack and pack.media else None,
            }
        )
    return rows


def prune(runs_dir: Path, now: datetime | None = None) -> list[str]:
    """Delete runs beyond the rolling buffer; returns the removed run ids."""
    now = now or datetime.now(UTC)
    removed = []
    for position, trial in enumerate(all_trials(runs_dir)):
        if position >= MAX_RUNS or now - trial.created_at > MAX_AGE:
            shutil.rmtree(runs_dir / trial.run_id, ignore_errors=True)
            removed.append(trial.run_id)
    return removed


class RateLimiter:
    """At most ``limit`` events per key in any rolling hour."""

    def __init__(self, limit: int = RUNS_PER_HOUR) -> None:
        self.limit = limit
        self.events: defaultdict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        recent = self.events[key]
        while recent and now - recent[0] > 3600:  # noqa: PLR2004 - one hour, in seconds
            recent.popleft()
        if len(recent) >= self.limit:
            return False
        recent.append(now)
        return True
