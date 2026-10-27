# Off My Shelf web app. Build: docker build -t off-my-shelf .
# Run:   docker run --rm -p 7860:7860 --env-file .env off-my-shelf   -> http://localhost:7860
# The same image runs on Hugging Face Spaces (Docker SDK, port 7860).

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

FROM python:3.12-slim-bookworm
# OpenCV (headless) needs GLib for video decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMS_HOST=0.0.0.0 \
    OMS_PORT=7860 \
    OMS_CACHE_DIR=/home/app/.cache/off-my-shelf \
    OMS_RUNS_DIR=/home/app/runs
EXPOSE 7860
CMD ["off-my-shelf", "serve"]
