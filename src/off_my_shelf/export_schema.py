"""Write the contract's JSON Schemas to disk.

Usage (from the repo root): ``uv run python -m off_my_shelf.export_schema [OUT_DIR]``

- ``listing_pack.schema.json``: a full run, as written by the pipeline and read by the frontend.
- ``item_analysis.schema.json``: the structured-output format sent to Muse Spark.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from off_my_shelf.schema import ItemAnalysis, ListingPack

DEFAULT_OUT_DIR = Path("schemas")
SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "listing_pack.schema.json": ListingPack,
    "item_analysis.schema.json": ItemAnalysis,
}


def render_schema(model: type[BaseModel]) -> str:
    """Serialise one model's JSON Schema deterministically, with a trailing newline."""
    schema: dict[str, Any] = model.model_json_schema()
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def write_schemas(out_dir: Path) -> list[Path]:
    """Write every contract schema into ``out_dir`` and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, model in SCHEMA_MODELS.items():
        path = out_dir / filename
        path.write_text(render_schema(model), encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main(argv: list[str]) -> None:
    out_dir = Path(argv[0]) if argv else DEFAULT_OUT_DIR
    for path in write_schemas(out_dir):
        print(f"wrote {path.as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1:])
