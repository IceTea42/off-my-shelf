# AGENTS.md

Instructions for coding agents working in this repository. The owner keeps a private living plan;
re-read it at the start of every brick and before any decision that touches another lane.

## Working rules

1. **Plan first.** Before any work, give a list of testable work bricks and wait for approval.
2. **One brick at a time.** Do the bricks in order; at the end of each, say exactly how to test it
   and wait for the result before starting the next.
3. **Autonomous inside a brick.** Decide routine details yourself; ask only when a decision is the
   owner's.
4. **Never commit or push.** Make the edits locally, then give the draft commit message and the
   exact commands to commit and push.
5. **Commit message format:** `<type>(<scope>): <summary>`, first line at most 72 characters,
   imperative mood. Add a body only when really needed (about 5% of commits), at most 200
   characters.
   - `type`: feat, fix, docs, chore, refactor, test, ci, build, improve.
   - `scope`: 1 or 2 kebab-case words naming the module or file, e.g. `sam-client`, `readme`.
   - Example: `feat(sam-client): cache masks by input hash`
6. **Ask before anything public.** Draft anything publicly visible (pushed content, GitHub
   settings, posts, messages, submission forms), ask for validation, act only after permission.
7. **Best coding practices.** Typed, tested, lint-clean code; small focused functions; no dead code
   or unused variables.
8. **No assistant branding.** Never name the assistant or its vendor in code, docs or commits;
   agent instructions live in this file.
9. **Cross-platform.** Code runs on Windows and Linux: `pathlib` paths, UTF-8 file I/O, no
   shell-only scripts, LF line endings. CI tests both.
10. **Rules can grow.** Propose new rules to the owner; add them only once validated.
11. **Re-read the plan.** At the start of every brick, before cross-lane decisions (schema,
    interfaces, dependencies, repo layout) and before proposing anything public.

## Repository conventions

- Python 3.12, `uv` project, src layout: package code in `src/off_my_shelf/`, tests in `tests/`.
- Add dependencies with `uv add <pkg>` (dev tools: `uv add --dev <pkg>`); commit `uv.lock`.
- Secrets live in `.env` (template: `.env.example`), never in code or git.
- Cache every model API result on disk under `OMS_CACHE_DIR`, keyed by input hash; develop on
  cached or mocked results and hit live endpoints only for smoke tests and final runs.
- The listing schema in `src/off_my_shelf/schema.py` is the contract between lanes; change it only
  with the owner's approval and regenerate `schemas/` afterwards.
- Raw footage stays out of git (`data/raw/`, size-capped by pre-commit).

## Checks (must pass before every commit)

```
uv sync
uv run pre-commit install        # once per clone
uv run pre-commit run --all-files
uv run pytest
```

Stage every new file before committing: hooks and CI see what is committed, not your working tree.
