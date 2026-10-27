"""Cost and latency of a listing pack, per pipeline step and model, from each item's calls."""

import statistics
from dataclasses import dataclass
from itertools import groupby

from off_my_shelf.schema import ListingPack, ModelCall


@dataclass(frozen=True, slots=True)
class StepCost:
    step: str
    model: str
    calls: int
    live_calls: int
    cost_usd: float
    cost_per_item_usd: float
    median_latency_ms: int | None


def live_cost(pack: ListingPack) -> float:
    """What the pack's live (not cached) calls cost at list price."""
    return sum(c.cost_usd or 0.0 for item in pack.items for c in item.calls if not c.cached)


def step_costs(pack: ListingPack) -> list[StepCost]:
    """One row per (step, model); latency from live calls only (cache hits have none)."""
    items = max(len(pack.items), 1)
    calls = sorted(
        (call for item in pack.items for call in item.calls), key=lambda c: (c.step, c.model)
    )
    rows = []
    for (step, model), group in groupby(calls, key=lambda c: (c.step, c.model)):
        members: list[ModelCall] = list(group)
        cost = sum(c.cost_usd or 0.0 for c in members)
        latencies = [c.latency_ms for c in members if c.latency_ms is not None]
        rows.append(
            StepCost(
                step=step.value,
                model=model,
                calls=len(members),
                live_calls=sum(not c.cached for c in members),
                cost_usd=cost,
                cost_per_item_usd=cost / items,
                median_latency_ms=round(statistics.median(latencies)) if latencies else None,
            )
        )
    return rows


def markdown_table(pack: ListingPack) -> str:
    """The README / tech report table for one pack."""
    rows = step_costs(pack)
    lines = [
        "| Step | Model | Calls (live) | Cost per item | Median latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        latency = f"{row.median_latency_ms / 1000:.1f} s" if row.median_latency_ms else "cached"
        lines.append(
            f"| {row.step} | {row.model} | {row.calls} ({row.live_calls}) "
            f"| ${row.cost_per_item_usd:.4f} | {latency} |"
        )
    total = sum(row.cost_usd for row in rows)
    lines.append(f"\n{len(pack.items)} items, list cost ${total:.4f} in total.")
    return "\n".join(lines)
