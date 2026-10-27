"""Client for the Meta Model API, which speaks the OpenAI wire protocol."""

from openai import AsyncOpenAI

from off_my_shelf.config import Settings


class MissingApiKeyError(RuntimeError):
    """Raised when a live call is attempted without META_API_KEY."""

    def __init__(self) -> None:
        super().__init__(
            "META_API_KEY is not set: copy .env.example to .env and fill it in, "
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
