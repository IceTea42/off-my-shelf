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
| Web app (voice goals, spending cap) | `web.py`, `static/index.html` | all |
| Cost and latency report | `report.py` | — |
| Translations and lots | `spark.py`, `agent.py`, `web.py` | Muse Spark |
| Video ads from real shots | `video_ad.py`, `fonts/` (DejaVu, free license) | Muse Spark |
| Android app (PWA: camera, mic, share) | `static/manifest.webmanifest`, `static/sw.js` | — |
| Hugging Face Space deploy | `scripts/deploy_space.py`, `.github/workflows/ci.yml` | — |
| Docs (deploy, phone, tech report) | `docs/` | — |
| Mini eval | `evaluate.py`, `data/eval/` | Muse Spark, SAM 3.1 |
| Repo tooling (uv, pre-commit, CI on Windows + Ubuntu, Docker) | `pyproject.toml`, `.github/`, `Dockerfile` | — |

Third-party libraries are used as published (see `pyproject.toml`); the DejaVu fonts are
bundled under their own license (`src/off_my_shelf/fonts/LICENSE-DejaVu.txt`). No code, data or
assets from any employer or earlier project are included.
