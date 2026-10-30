"""Clients for the Meta Model API and for Muse Spark Open, both on the OpenAI wire protocol."""

from openai import APIStatusError, AsyncOpenAI
from pydantic import SecretStr

from off_my_shelf.config import Settings


class MissingApiKeyError(RuntimeError):
    """Raised when a live call is attempted without the needed key."""

    def __init__(self, variable: str = "META_API_KEY") -> None:
        super().__init__(
            f"{variable} is not set: copy .env.example to .env and fill it in, "
            "or run in offline mode."
        )


def make_client(settings: Settings) -> AsyncOpenAI:
    """An async client bound to the Meta endpoint. Retries cover 429 and 5xx responses."""
    if settings.meta_api_key is None:
        raise MissingApiKeyError
    return AsyncOpenAI(
        api_key=settings.meta_api_key.get_secret_value(),
        base_url=settings.meta_base_url,
        timeout=settings.request_timeout_s,
        max_retries=3,
    )


def open_weight_endpoints(settings: Settings) -> list[tuple[str, str, SecretStr]]:
    """Partner endpoints serving Muse Spark Open, in fallback order: (name, base URL, key)."""
    endpoints = []
    if settings.baseten_api_key is not None:
        endpoints.append(("baseten", settings.baseten_base_url, settings.baseten_api_key))
    if settings.modal_spark_open_url and settings.modal_spark_open_key is not None:
        endpoints.append(("modal", settings.modal_spark_open_url, settings.modal_spark_open_key))
    return endpoints


def make_open_weights_client(settings: Settings, base_url: str, key: SecretStr) -> AsyncOpenAI:
    """A client for one partner endpoint; one retry, since the next provider is the fallback."""
    return AsyncOpenAI(
        api_key=key.get_secret_value(),
        base_url=base_url,
        timeout=settings.request_timeout_s,
        max_retries=1,
    )


def describe_error(error: Exception) -> str:
    """What went wrong, with the API's own status, path and error body when there is one."""
    if isinstance(error, APIStatusError):
        request_id = error.response.headers.get("x-request-id") or error.request_id or "none"
        body = error.body if error.body is not None else error.response.text
        return (
            f"HTTP {error.status_code} from {error.request.method} {error.request.url.path}: "
            f"{body} (request id {request_id})"
        )
    return f"{type(error).__name__}: {error}"
