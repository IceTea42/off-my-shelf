"""Traces of every model call and agent step in Pydantic Logfire, when LOGFIRE_TOKEN is set in .env.

Without a token nothing leaves the machine; with one, the Logfire dashboard shows each run:
the Meta, Baseten and Modal calls with their tokens and latency, and each agent step with the
tool it called, its arguments and whether its checks let it through.
"""

from functools import cache

import logfire
from fastapi import FastAPI
from pydantic import SecretStr


@cache
def setup_tracing(token: SecretStr | None) -> None:
    logfire.configure(
        token=token.get_secret_value() if token else None,
        send_to_logfire="if-token-present",
        service_name="off-my-shelf",
        console=False,
    )
    logfire.instrument_openai()  # every OpenAI-compatible client: Meta, Baseten, Modal


def trace_app(app: FastAPI, token: SecretStr | None) -> None:
    setup_tracing(token)
    logfire.instrument_fastapi(app)
