"""Load listing packs from disk, including the bundled mock pack other lanes build against."""

from pathlib import Path

from off_my_shelf.schema import ListingPack

PACK_FILENAME = "listing_pack.json"
MOCK_PACK_DIR = Path(__file__).parent / "mock_data"


def load_pack(pack_dir: Path) -> ListingPack:
    """Read and validate ``listing_pack.json`` from a pack directory.

    Paths inside the pack are relative to ``pack_dir``: resolve them with ``pack_dir / path``.
    """
    raw = (pack_dir / PACK_FILENAME).read_text(encoding="utf-8")
    return ListingPack.model_validate_json(raw)


def save_pack(pack_dir: Path, pack: ListingPack) -> None:
    """Write ``listing_pack.json`` (LF endings, UTF-8) into a pack directory."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / PACK_FILENAME).write_text(
        pack.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def load_mock_pack() -> ListingPack:
    """The 5-item mock pack: all triage decisions, both pricing modes, FR/EN/TR listings."""
    return load_pack(MOCK_PACK_DIR)
