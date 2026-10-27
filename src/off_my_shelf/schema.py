"""Listing contract shared by every lane: pipeline, frontend, eval and tech report.

Two layers:
- ``ItemAnalysis`` is what Muse Spark returns as structured output for one item crop.
- ``Item`` wraps it with what the pipeline adds (detection, media, agent outputs, provenance),
  and ``ListingPack`` is one run over a set of photos or videos.

File paths are POSIX strings relative to the pack directory, so a pack moves between
Windows and Linux unchanged.
"""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final = "1.0"


def _check_relative_posix(path: str) -> str:
    if "\\" in path:
        raise ValueError("use '/' separators, not '\\'")
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts or ":" in posix.parts[0]:
        raise ValueError("must be a relative path inside the pack directory")
    return path


RelPath = Annotated[str, Field(min_length=1), AfterValidator(_check_relative_posix)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
ShortText = Annotated[str, Field(min_length=1, max_length=300)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


# --- Enums -------------------------------------------------------------------------------


class Category(StrEnum):
    ELECTRONICS = "electronics"
    BOOKS_MEDIA = "books_media"
    HOME_KITCHEN = "home_kitchen"
    FURNITURE = "furniture"
    CLOTHING = "clothing"
    KIDS_TOYS = "kids_toys"
    SPORTS_OUTDOOR = "sports_outdoor"
    TOOLS_DIY = "tools_diy"
    OTHER = "other"


class ConditionGrade(StrEnum):
    NEW = "new"
    LIKE_NEW = "like_new"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BROKEN = "broken"


class Language(StrEnum):
    FR = "fr"
    EN = "en"
    TR = "tr"


class TriageDecision(StrEnum):
    SELL = "sell"
    DONATE = "donate"
    RECYCLE = "recycle"


class PricingMode(StrEnum):
    SELL_FAST = "sell_fast"
    MAX_VALUE = "max_value"


class PipelineStep(StrEnum):
    SEGMENT = "segment"
    ANALYZE = "analyze"
    PRODUCT_SHOT = "product_shot"
    TRANSCRIBE = "transcribe"
    TRIAGE = "triage"
    BUYER_REPLY = "buyer_reply"
    AD = "ad"


# --- Muse Spark structured output --------------------------------------------------------


class Identification(_Model):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    brand: Annotated[str, Field(max_length=60)] | None = None
    model: Annotated[str, Field(max_length=60)] | None = None
    category: Category
    confidence: Unit = Field(description="Model's confidence in the identification, 0 to 1.")


class Condition(_Model):
    grade: ConditionGrade
    defects: Annotated[list[ShortText], Field(max_length=10)] = []


class PriceEstimate(_Model):
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] = "EUR"
    low: Annotated[float, Field(ge=0.0)]
    suggested: Annotated[float, Field(ge=0.0)]
    high: Annotated[float, Field(ge=0.0)]
    rationale: ShortText

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.low <= self.suggested <= self.high:
            raise ValueError("price must satisfy low <= suggested <= high")
        return self


class ListingText(_Model):
    language: Language = Language.EN
    title: Annotated[str, Field(min_length=3, max_length=80)]
    description: Annotated[str, Field(min_length=1, max_length=2000)]
    tags: Annotated[list[Annotated[str, Field(min_length=1, max_length=30)]], Field(max_length=10)]


class Triage(_Model):
    decision: TriageDecision
    reason: ShortText


class ItemAnalysis(_Model):
    """One item as analysed by Muse Spark; its JSON Schema is the structured-output format."""

    identification: Identification
    condition: Condition
    price: PriceEstimate
    listing: ListingText
    triage: Triage


# --- Pipeline additions ------------------------------------------------------------------


class BoundingBox(_Model):
    """Normalised to the source image or frame size: (0, 0) is top-left, (1, 1) bottom-right."""

    x0: Unit
    y0: Unit
    x1: Unit
    y1: Unit

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("bounding box needs x0 < x1 and y0 < y1")
        return self


class SourceRef(_Model):
    media: RelPath
    frame_index: Annotated[int, Field(ge=0)] | None = None
    timestamp_s: Annotated[float, Field(ge=0.0)] | None = None


class Detection(_Model):
    """SAM 3.1 output for one item."""

    source: SourceRef
    bbox: BoundingBox
    score: Unit
    crop_path: RelPath
    mask_path: RelPath | None = None


class ModelCall(_Model):
    """Provenance of one model call: feeds the cost and latency table."""

    step: PipelineStep
    model: Annotated[str, Field(min_length=1)]
    cache_key: Annotated[str, Field(min_length=1)]
    cached: bool = False
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0.0)] | None = None
    latency_ms: Annotated[int, Field(ge=0)] | None = None


class BuyerReply(_Model):
    """Sell-me-softly draft: never sent until a human approves it."""

    buyer_message: Annotated[str, Field(min_length=1, max_length=1000)]
    draft_reply: Annotated[str, Field(min_length=1, max_length=1000)]
    approved: bool = False


class Item(_Model):
    item_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
    detection: Detection
    analysis: ItemAnalysis
    pricing_mode: PricingMode | None = None
    product_shots: list[RelPath] = []
    voice_note: Annotated[str, Field(max_length=2000)] | None = None
    buyer_replies: list[BuyerReply] = []
    calls: list[ModelCall] = []


class ListingPack(_Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: Annotated[str, Field(min_length=1, max_length=64)]
    created_at: AwareDatetime
    media: Annotated[list[RelPath], Field(min_length=1)]
    items: list[Item]
    warnings: list[str] = Field(
        default_factory=list, description="Inputs or items skipped, and why, for the person"
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("item_id values must be unique")
        unknown = {item.detection.source.media for item in self.items} - set(self.media)
        if unknown:
            raise ValueError(f"items reference media not listed in 'media': {sorted(unknown)}")
        return self


__all__ = [
    "SCHEMA_VERSION",
    "BoundingBox",
    "BuyerReply",
    "Category",
    "Condition",
    "ConditionGrade",
    "Detection",
    "Identification",
    "Item",
    "ItemAnalysis",
    "Language",
    "ListingPack",
    "ListingText",
    "ModelCall",
    "PipelineStep",
    "PriceEstimate",
    "PricingMode",
    "SourceRef",
    "Triage",
    "TriageDecision",
]
