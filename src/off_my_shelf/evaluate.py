"""Mini eval: run the pipeline on photos of items with known resale prices and score it.

Dataset: a CSV with one row per photo (one item per photo), columns
``image,name,brand,price_eur`` where ``image`` is relative to the CSV and ``brand`` may be
empty. Metrics:
- identification accuracy: predicted name+brand shares at least half its words with the truth;
- price error: median absolute percentage error of the suggested price, and how often the true
  price falls inside the predicted low..high range;
- brand hallucination rate: a brand was asserted but the truth has none or a different one.
"""

import csv
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from off_my_shelf.config import Settings
from off_my_shelf.imaging import encode
from off_my_shelf.mocks import load_pack
from off_my_shelf.pipeline import Pipeline, Services
from off_my_shelf.schema import Item

WORD = re.compile(r"[\w']+", re.UNICODE)
MATCH_THRESHOLD = 0.5
EVAL_MAX_SIDE = 1280


@dataclass(frozen=True, slots=True)
class Truth:
    image: Path
    name: str
    brand: str | None
    price_eur: float


@dataclass(frozen=True, slots=True)
class Score:
    image: str
    predicted: str
    identified: bool
    price_error: float | None
    in_range: bool | None
    brand_hallucinated: bool
    cost_usd: float


def load_truth(csv_path: Path) -> list[Truth]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Truth(
            image=csv_path.parent / row["image"],
            name=row["name"].strip(),
            brand=(row.get("brand") or "").strip() or None,
            price_eur=float(row["price_eur"]),
        )
        for row in rows
        if row["price_eur"].strip()  # rows still to fill in are skipped
    ]


CSV_FIELDS = ["image", "name", "brand", "price_eur"]


def add_pack_to_eval(pack_dir: Path, csv_path: Path) -> list[str]:
    """Add each item crop of a real run to the eval set, with name, brand and price left
    blank for a person to fill in from real ads. Returns the model's names, as a reminder."""
    pack = load_pack(pack_dir)
    photos = csv_path.parent / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    hints = []
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, CSV_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        for item in pack.items:
            name = f"{pack.run_id}-{item.item_id}.jpg"
            with Image.open(pack_dir / item.detection.crop_path) as crop:
                image = crop.convert("RGB")
            image.thumbnail((EVAL_MAX_SIDE, EVAL_MAX_SIDE))
            (photos / name).write_bytes(encode(image, quality=85))
            writer.writerow({"image": f"photos/{name}", "name": "", "brand": "", "price_eur": ""})
            hints.append(f"photos/{name}: model says {item.analysis.identification.name}")
    return hints


def _words(text: str) -> set[str]:
    return {w.casefold() for w in WORD.findall(text)}


def name_matches(predicted: str, truth: str) -> bool:
    """At least half of the truth's words appear in the prediction."""
    expected = _words(truth)
    return bool(expected) and len(expected & _words(predicted)) / len(expected) >= MATCH_THRESHOLD


def score_item(truth: Truth, item: Item | None) -> Score:
    if item is None:
        return Score(truth.image.name, "", False, None, None, False, 0.0)
    ident = item.analysis.identification
    predicted = " ".join(filter(None, [ident.brand, ident.name]))
    price = item.analysis.price
    asserted = (ident.brand or "").casefold()
    hallucinated = bool(asserted) and asserted != (truth.brand or "").casefold()
    return Score(
        image=truth.image.name,
        predicted=predicted,
        identified=name_matches(predicted, " ".join(filter(None, [truth.brand, truth.name]))),
        price_error=abs(price.suggested - truth.price_eur) / truth.price_eur,
        in_range=price.low <= truth.price_eur <= price.high,
        brand_hallucinated=hallucinated,
        cost_usd=sum(call.cost_usd or 0.0 for call in item.calls),
    )


def summarise(scores: list[Score]) -> dict[str, float | int | None]:
    errors = [s.price_error for s in scores if s.price_error is not None]
    ranged = [s.in_range for s in scores if s.in_range is not None]
    count = len(scores) or 1
    return {
        "items": len(scores),
        "identification_accuracy": sum(s.identified for s in scores) / count,
        "median_price_error": statistics.median(errors) if errors else None,
        "true_price_in_range": sum(ranged) / len(ranged) if ranged else None,
        "brand_hallucination_rate": sum(s.brand_hallucinated for s in scores) / count,
        "mean_cost_usd": sum(s.cost_usd for s in scores) / count,
    }


async def run_eval(
    services: Services, settings: Settings, csv_path: Path, out_dir: Path
) -> dict[str, float | int | None]:
    """Run every photo, keep its most confident item, write ``report.md`` and ``scores.csv``."""
    scores = []
    for number, truth in enumerate(load_truth(csv_path), start=1):
        pack = await Pipeline(services, settings).run(
            [truth.image], out_dir / "packs" / f"{number:03d}"
        )
        best = max(pack.items, key=lambda i: i.analysis.identification.confidence, default=None)
        scores.append(score_item(truth, best))
    summary = summarise(scores)
    _write_reports(scores, summary, out_dir)
    return summary


def _write_reports(
    scores: list[Score], summary: dict[str, float | int | None], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Score.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(s) for s in scores)
    lines = ["# Off My Shelf mini eval", "", "| Metric | Value |", "| --- | --- |"]
    for metric, value in summary.items():
        shown = "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else value
        lines.append(f"| {metric.replace('_', ' ')} | {shown} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
