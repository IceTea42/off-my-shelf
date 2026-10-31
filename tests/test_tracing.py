"""The model-call logs the Logfire dashboard is built on."""

from logfire.testing import CaptureLogfire

from off_my_shelf.schema import ModelCall, PipelineStep
from off_my_shelf.tracing import record_calls


def test_each_call_is_logged_with_its_cost(capfire: CaptureLogfire) -> None:
    live = ModelCall(
        step=PipelineStep.ANALYZE, model="m", cache_key="k", cost_usd=0.01, provider="baseten"
    )
    cached = ModelCall(step=PipelineStep.SEGMENT, model="s", cache_key="k", cached=True, cost_usd=1)
    record_calls([live, cached], run="r1", item="item-001")
    logs = [span["attributes"] for span in capfire.exporter.exported_spans_as_dict()]
    assert [(a["step"], a["provider"], a["cost_usd"]) for a in logs] == [
        ("analyze", "baseten", 0.01),
        ("segment", "meta", 0.0),  # a cache hit costs nothing
    ]
    assert logs[0]["run"] == "r1" and logs[0]["item"] == "item-001"
