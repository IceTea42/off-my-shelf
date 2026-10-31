"""Traces of every model call and agent step in Pydantic Logfire, when LOGFIRE_TOKEN is set in .env.

Without a token nothing leaves the machine; with one, the Logfire dashboard shows each run:
the Meta, Baseten and Modal calls with their tokens and latency, and each agent step with the
tool it called, its arguments and whether its checks let it through.
"""

from collections.abc import Iterable
from functools import cache

import logfire
from fastapi import FastAPI
from pydantic import SecretStr

from off_my_shelf.schema import ModelCall


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


def record_calls(calls: Iterable[ModelCall], *, run: str, item: str | None = None) -> None:
    """One ``model call`` log per call, with the fields the Logfire dashboard charts: step,
    model, provider, list cost, latency, tokens and whether the cache served it."""
    for call in calls:
        logfire.info(
            "model call {step} {model}",
            step=call.step.value,
            model=call.model,
            provider=call.provider or "meta",
            cost_usd=0.0 if call.cached else call.cost_usd or 0.0,
            cached=call.cached,
            latency_ms=call.latency_ms,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            run=run,
            item=item,
        )
