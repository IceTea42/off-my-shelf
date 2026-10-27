"""Listing pack export: a self-contained HTML page per pack, plain-text listings and a zip."""

import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, PackageLoader, select_autoescape

from off_my_shelf.mocks import load_pack
from off_my_shelf.schema import ListingPack, TriageDecision

INDEX_FILENAME = "index.html"
FAVICON = quote((Path(__file__).parent / "static" / "favicon.svg").read_text(encoding="utf-8"))
LISTINGS_DIRNAME = "listings"

_env = Environment(
    loader=PackageLoader("off_my_shelf", "templates"),
    autoescape=select_autoescape(["html"]),
    keep_trailing_newline=True,
)


def render_html(pack: ListingPack) -> str:
    counts = Counter(item.analysis.triage.decision.value for item in pack.items)
    selling = [i for i in pack.items if i.analysis.triage.decision is TriageDecision.SELL]
    currencies = {i.analysis.price.currency for i in selling} or {"EUR"}
    return _env.get_template("pack.html").render(
        pack=pack,
        counts={d.value: counts.get(d.value, 0) for d in TriageDecision},
        total_value=sum(i.analysis.price.suggested for i in selling),
        currency="/".join(sorted(currencies)),
        lang=pack.items[0].analysis.listing.language.value if pack.items else "en",
        favicon=FAVICON,
    )


def listing_text(pack: ListingPack) -> dict[str, str]:
    """Ready-to-paste text per item, keyed by item id."""
    texts = {}
    for item in pack.items:
        a = item.analysis
        texts[item.item_id] = (
            f"{a.listing.title}\n\n{a.listing.description}\n\n"
            f"Price: {a.price.suggested:.0f} {a.price.currency}\n"
            f"Tags: {', '.join(a.listing.tags)}\n"
        )
    return texts


def export_pack(pack_dir: Path) -> Path:
    """Write ``index.html`` and ``listings/*.txt`` into the pack, then zip it. Returns the zip."""
    pack = load_pack(pack_dir)
    (pack_dir / INDEX_FILENAME).write_text(render_html(pack), encoding="utf-8", newline="\n")
    listings = pack_dir / LISTINGS_DIRNAME
    listings.mkdir(exist_ok=True)
    for item_id, text in listing_text(pack).items():
        (listings / f"{item_id}.txt").write_text(text, encoding="utf-8", newline="\n")

    archive = pack_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pack_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(pack_dir).as_posix())
    return archive
