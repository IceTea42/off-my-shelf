"""Runtime settings, read from the environment and an optional ``.env`` file.

Every field has a safe default, so the offline path (mocks, cached results) runs with no
``.env`` at all. Only live model calls need ``META_API_KEY``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from off_my_shelf.schema import Language


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="OMS_", extra="ignore"
    )

    # Meta Model API. MODEL_API_KEY is the name used in Meta's own docs.
    meta_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("META_API_KEY", "MODEL_API_KEY")
    )
    meta_base_url: str = "https://api.meta.ai/v1"
    spark_model: str = "muse-spark-1.3"
    image_model: str = "muse-image-1.0"
    sam_model: str = "sam-3.1"
    transcribe_model: str = "muse-voice-transcribe-1.0"

    # Listings
    default_language: Language = Language.EN

    # Disk cache for model results, keyed by input hash
    cache_dir: Path = Path(".cache")

    # Bounds that keep each call small and cheap
    max_image_side_px: int = Field(default=1280, ge=256)
    # Reasoning tokens count toward the output budget, so this leaves room for thinking.
    max_output_tokens: int = Field(default=4000, ge=256)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=4, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()
