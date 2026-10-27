"""List prices of the Meta models used (Standard mode, USD), for the cost-per-item table.

Source: Meta Model API pricing page, October 2026. Update here if prices change.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TokenPrice:
    input_per_m: float
    output_per_m: float


TOKEN_PRICES: Final[dict[str, TokenPrice]] = {
    "muse-spark-1.3": TokenPrice(input_per_m=1.25, output_per_m=4.25),
    "muse-spark-1.2": TokenPrice(input_per_m=1.25, output_per_m=4.25),
    "muse-wonder-1.0": TokenPrice(input_per_m=4.00, output_per_m=20.00),
}
IMAGE_GENERATION_USD: Final = 0.01  # muse-image-1.0, per returned image
SAM_IMAGE_USD: Final = 2.50 / 1000  # sam-3.1, per segmented image
TRANSCRIBE_USD_PER_HOUR: Final = 0.18  # muse-voice-transcribe-1.0


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one token-priced call, or None for a model without a listed price."""
    price = TOKEN_PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000


def transcribe_cost(audio_duration_ms: int) -> float:
    return TRANSCRIBE_USD_PER_HOUR * audio_duration_ms / 3_600_000
