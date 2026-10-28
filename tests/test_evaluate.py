"""Mini eval: dataset loading, matching rules, metrics and reports on a fake client."""

from pathlib import Path

import pytest
from PIL import Image

from fakes import SAM_SIZE, FakeMetaClient
from off_my_shelf.cache import DiskCache
from off_my_shelf.cli import main
from off_my_shelf.config import Settings
from off_my_shelf.evaluate import (
    Truth,
    add_pack_to_eval,
    load_truth,
    name_matches,
    run_eval,
    score_item,
    summarise,
)
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack
from off_my_shelf.pipeline import Services
from off_my_shelf.sam import SamSegmenter
from off_my_shelf.spark import SparkAnalyzer

LAMP = load_mock_pack().items[0]  # "Lampe de bureau articulée", 8..18, suggested 12


def _dataset(folder: Path, price: float = 10.0) -> Path:
    (folder / "photos").mkdir(parents=True)
    Image.new("RGB", SAM_SIZE, "white").save(folder / "photos" / "lamp.jpg")
    csv_path = folder / "items.csv"
    csv_path.write_text(
        f"image,name,brand,price_eur\nphotos/lamp.jpg,lampe de bureau,,{price}\n",
        encoding="utf-8",
    )
    return csv_path


def test_name_matching() -> None:
    assert name_matches("Lampe de bureau articulée", "lampe bureau")
    assert not name_matches("Bouilloire", "lampe de bureau")
    assert not name_matches("anything", "")


def test_score_and_summary(tmp_path: Path) -> None:
    truth = Truth(tmp_path / "a.jpg", "lampe de bureau", None, 10.0)
    scored = score_item(truth, LAMP)
    assert scored.identified and scored.in_range
    assert scored.price_error == pytest.approx(0.2)
    assert not scored.brand_hallucinated
    branded = LAMP.model_copy(deep=True)
    branded.analysis.identification.brand = "Ikea"
    assert score_item(truth, branded).brand_hallucinated
    missing = score_item(truth, None)
    summary = summarise([scored, missing])
    assert summary["identification_accuracy"] == 0.5
    assert summary["median_price_error"] == pytest.approx(0.2)
    assert summary["true_price_in_range"] == 1.0


async def test_run_eval_writes_reports(tmp_path: Path) -> None:
    csv_path = _dataset(tmp_path / "data")
    assert load_truth(csv_path)[0].brand is None
    settings = Settings(
        _env_file=None, cache_dir=tmp_path / "cache", min_item_side_px=1, min_item_pixels=1
    )  # type: ignore[call-arg]
    fake = FakeMetaClient(["desk lamp"])
    cache = DiskCache(settings.cache_dir)
    services = Services(
        sam=SamSegmenter(fake, "sam-3.1", cache),  # type: ignore[arg-type]
        spark=SparkAnalyzer(fake, "muse-spark-1.3", cache, 4000),  # type: ignore[arg-type]
    )
    summary = await run_eval(services, settings, csv_path, tmp_path / "report")
    assert summary["items"] == 1 and summary["identification_accuracy"] == 1.0
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "| identification accuracy | 1.000 |" in report
    assert (tmp_path / "report" / "scores.csv").is_file()


def test_cli_eval_offline_needs_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = _dataset(tmp_path / "data")
    with pytest.raises(LookupError):
        main(["eval", str(csv_path), "--offline"])


def test_add_pack_to_eval_then_skip_unfilled_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "eval" / "items.csv"
    hints = add_pack_to_eval(MOCK_PACK_DIR, dataset)
    assert len(hints) == len(load_mock_pack().items)
    assert load_truth(dataset) == []  # nothing filled in yet
    rows = dataset.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "image,name,brand,price_eur"
    rows[1] = rows[1].replace(",,,", ",desk lamp,,12")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    [truth] = load_truth(dataset)
    assert truth.name == "desk lamp" and truth.price_eur == 12 and truth.image.is_file()
