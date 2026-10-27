"""Clients for the Meta Model API and for Muse Spark Open, both on the OpenAI wire protocol."""

from openai import AsyncOpenAI

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


def make_open_weights_client(settings: Settings) -> AsyncOpenAI:
    """An async client for Muse Spark Open on the configured partner endpoint."""
    if settings.spark_open_api_key is None:
        raise MissingApiKeyError("BASETEN_API_KEY (or OMS_SPARK_OPEN_API_KEY)")
    return AsyncOpenAI(
        api_key=settings.spark_open_api_key.get_secret_value(),
        base_url=settings.spark_open_base_url,
        timeout=settings.request_timeout_s,
        max_retries=3,
    )
