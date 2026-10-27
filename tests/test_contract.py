"""The listing contract: mock pack validity and exported JSON Schemas kept in sync."""

from pathlib import Path

import pytest

from off_my_shelf.export_schema import SCHEMA_MODELS, render_schema, write_schemas
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack
from off_my_shelf.schema import BoundingBox, PriceEstimate, SourceRef

REPO_SCHEMAS = Path(__file__).parents[1] / "schemas"


def test_mock_pack_loads_and_its_files_exist() -> None:
    pack = load_mock_pack()
    assert len(pack.items) == 5
    for item in pack.items:
        for rel in (item.detection.crop_path, *item.product_shots):
            assert (MOCK_PACK_DIR / rel).is_file(), rel


@pytest.mark.parametrize("filename", sorted(SCHEMA_MODELS))
def test_exported_schemas_are_up_to_date(filename: str) -> None:
    committed = (REPO_SCHEMAS / filename).read_text(encoding="utf-8")
    assert committed == render_schema(SCHEMA_MODELS[filename]), (
        "schema changed: run `uv run python -m off_my_shelf.export_schema`"
    )


def test_write_schemas(tmp_path: Path) -> None:
    assert sorted(p.name for p in write_schemas(tmp_path)) == sorted(SCHEMA_MODELS)


@pytest.mark.parametrize("bad", ["C:/x.png", "/abs.png", "../up.png", "a\\b.png"])
def test_paths_must_be_relative_posix(bad: str) -> None:
    with pytest.raises(ValueError, match=r"path|separators"):
        SourceRef(media=bad)


def test_value_rules() -> None:
    with pytest.raises(ValueError, match="low <= suggested"):
        PriceEstimate(low=10, suggested=5, high=20, rationale="x")
    with pytest.raises(ValueError, match="x0 < x1"):
        BoundingBox(x0=0.5, y0=0, x1=0.5, y1=1)
