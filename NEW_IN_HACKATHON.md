# Built during the hackathon

Everything in this repository was written during the Meta Global AI Developer Hackathon
(October 26 – November 4, 2026). There is no earlier codebase: the first commit is the empty
repository, and `git log --reverse` shows the work in the order it was done.

| Part | Files | Meta models |
| --- | --- | --- |
| Listing contract (schema + mock pack) | `src/off_my_shelf/schema.py`, `mocks.py`, `schemas/` | — |
| Core plumbing (settings, cache, costs, live smoke checks) | `config.py`, `cache.py`, `costs.py`, `smoke.py` | all |
| Segmentation client | `sam.py`, `imaging.py` | SAM 3.1 |
| Scene inventory and listings | `spark.py` | Muse Spark |
| Pipeline and CLI (photos, videos, keyframes) | `pipeline.py`, `media.py`, `cli.py` | Muse Spark, SAM 3.1 |
| Listing pack (product shots, voice notes, HTML/zip) | `product_shots.py`, `voice.py`, `export.py` | Muse Image, Voice Transcribe |
| Shelf agent (tool calls, checked actions, buyer replies) | `agent.py` | Muse Spark, Voice Transcribe |
| Web app | `web.py`, `static/index.html` | all |
| Mini eval | `evaluate.py`, `data/eval/` | Muse Spark, SAM 3.1 |
| Repo tooling (uv, pre-commit, CI on Windows + Ubuntu, Docker) | `pyproject.toml`, `.github/`, `Dockerfile` | — |

Third-party libraries are used as published (see `pyproject.toml`); no code, data or assets
from any employer or earlier project are included.
