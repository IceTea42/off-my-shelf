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

    # Open weights: the listing step runs on Muse Spark Open at each configured partner, in this
    # order (Baseten, then Modal); Muse Spark on the Meta Model API is the last fallback.
    spark_open_model: str = "meta-models/muse-spark-open-1026"
    baseten_base_url: str = "https://inference.baseten.co/v1"
    baseten_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("BASETEN_API_KEY", "OMS_BASETEN_API_KEY")
    )
    modal_spark_open_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MODAL_SPARK_OPEN_URL", "OMS_MODAL_SPARK_OPEN_URL"),
    )
    modal_spark_open_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MODAL_SPARK_OPEN_KEY", "OMS_MODAL_SPARK_OPEN_KEY"),
    )

    # 3D: SAM 3D Objects on a Modal GPU (scripts/modal_sam3d.py) for the item's 360-degree turn
    modal_sam3d_url: str | None = Field(
        default=None, validation_alias=AliasChoices("MODAL_SAM3D_URL", "OMS_MODAL_SAM3D_URL")
    )
    modal_sam3d_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MODAL_SAM3D_KEY", "OMS_MODAL_SAM3D_KEY")
    )

    # Traces in Pydantic Logfire (off when unset; read from .env like the other keys)
    logfire_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LOGFIRE_TOKEN", "OMS_LOGFIRE_TOKEN")
    )

    # Listings
    default_language: Language = Language.EN

    # Disk cache for model results, keyed by input hash
    cache_dir: Path = Path(".cache")

    # Web app: where uploads and packs of each run are kept
    runs_dir: Path = Path("packs/web")
    host: str = "127.0.0.1"
    port: int = 8000
    # Debug only: keep every voice note (upload, API WAV, outcome) under <cache_dir>/voice
    debug_voice: bool = False
    # Public deployments: refuse new live work once this much (list price, USD) is spent
    budget_usd: float | None = Field(default=None, gt=0)

    # Bounds that keep each call small and cheap
    max_image_side_px: int = Field(default=1280, ge=256)
    # Too little detail to list: photos/frames and items below these are skipped with a warning
    min_still_side_px: int = Field(default=320, ge=1)
    min_item_side_px: int = Field(default=40, ge=1)  # shortest side of the item's box
    min_item_pixels: int = Field(default=2500, ge=1)  # mask area, about 50 x 50
    # Reasoning tokens count toward the output budget, so this leaves room for thinking.
    max_output_tokens: int = Field(default=8000, ge=256)
    # Muse Spark reasoning depth: minimal | low | medium | high | xhigh ("none" is rejected)
    reasoning_effort: str | None = "low"
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()
