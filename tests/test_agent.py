"""Shelf agent: tool loop, verified actions, offline replay, CLI and web wiring."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from off_my_shelf import web
from off_my_shelf.agent import (
    AgentOutcome,
    AgentRequest,
    AgentStepLimitError,
    ShelfAgent,
    asking_price,
    check_buyer_reply,
    run_agent,
)
from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.config import Settings
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack, load_pack, save_pack
from off_my_shelf.schema import BuyerReply, PricingMode, TriageDecision

PACK = load_mock_pack()
FIRST = PACK.items[0]  # lamp, low 8, high 18


def _tool(name: str, call_id: str, **args: object) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def _reply(*tool_calls: SimpleNamespace, content: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls) or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=500, completion_tokens=100),
    )


class _ScriptedModel:
    """Returns scripted replies in order and records what the agent sent each step."""

    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = replies
        self.sent: list[list[dict[str, Any]]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        self.sent.append(json.loads(json.dumps(kwargs["messages"])))
        return self.replies[len(self.sent) - 1]


def _script() -> list[SimpleNamespace]:
    item = FIRST.item_id
    return [
        _reply(_tool("list_items", "c1")),
        _reply(
            _tool("set_pricing_mode", "c2", item_id=item, mode="sell_fast", reason="moving"),
            _tool("set_triage", "c3", item_id="item-002", decision="donate", reason="low value"),
            _tool("draft_buyer_reply", "c4", item_id=item, buyer_message="5€?", reply="OK 5 €"),
        ),
        _reply(
            _tool(
                "draft_buyer_reply",
                "c5",
                item_id=item,
                buyer_message="5€?",
                reply="Thanks! The lowest I can do is 9 €, collection in Paris.",
            ),
            _tool("set_triage", "c6", item_id="nope", decision="sell", reason="x"),
        ),
        _reply(_tool("finish", "c7", summary="Priced the lamp to sell fast and drafted a reply.")),
    ]


def test_asking_price_and_reply_checks() -> None:
    assert asking_price(FIRST, PricingMode.SELL_FAST) == 10
    assert asking_price(FIRST, PricingMode.MAX_VALUE) == 16
    assert check_buyer_reply(FIRST, "Yes, 9 € is fine") is None
    assert "below" in (check_buyer_reply(FIRST, "ok for 5€") or "")
    assert "marketplace" in (check_buyer_reply(FIRST, "text me on 06 12 34 56 78") or "")
    assert "marketplace" in (check_buyer_reply(FIRST, "pay by PayPal please") or "")


async def test_agent_applies_verified_actions(tmp_path: Path) -> None:
    model = _ScriptedModel(_script())
    agent = ShelfAgent(model, "muse-spark-1.3", DiskCache(tmp_path))  # type: ignore[arg-type]
    result = await agent.run(PACK, "Moving Friday, sell fast", {FIRST.item_id: "5€?"})

    lamp = next(i for i in result.pack.items if i.item_id == FIRST.item_id)
    assert lamp.pricing_mode is PricingMode.SELL_FAST
    assert lamp.analysis.price.suggested == 10
    before = len(FIRST.buyer_replies)
    assert len(lamp.buyer_replies) == before + 1  # the rejected draft was never saved
    assert lamp.buyer_replies[-1].draft_reply == (
        "Thanks! The lowest I can do is 9 €, collection in Paris."
    )
    assert lamp.buyer_replies[-1].approved is False
    second = next(i for i in result.pack.items if i.item_id == "item-002")
    assert second.analysis.triage.decision is TriageDecision.DONATE
    assert result.summary.startswith("Priced the lamp")
    assert len(result.calls) == 4 and result.calls[0].cost_usd is not None

    # The model saw the rejected reply and the unknown item, so it could correct itself.
    tool_results = [m["content"] for m in model.sent[2] if m["role"] == "tool"]
    assert any(r.startswith("rejected: 5 is below") for r in tool_results)
    final_results = [m["content"] for m in model.sent[3] if m["role"] == "tool"]
    assert any("unknown item_id" in r for r in final_results)
    assert "Buyer message about item-001" in model.sent[0][1]["content"]
    assert PACK.items[0].analysis.price.suggested == 12  # the input pack is untouched

    replay = ShelfAgent(None, "muse-spark-1.3", DiskCache(tmp_path))
    again = await replay.run(PACK, "Moving Friday, sell fast", {FIRST.item_id: "5€?"})
    assert again.pack == result.pack and all(c.cached for c in again.calls)


async def test_agent_step_limit(tmp_path: Path) -> None:
    looping = _ScriptedModel([_reply(_tool("list_items", f"c{n}")) for n in range(20)])
    agent = ShelfAgent(looping, "muse-spark-1.3", DiskCache(tmp_path))  # type: ignore[arg-type]
    with pytest.raises(AgentStepLimitError):
        await agent.run(PACK, "loop forever")


async def test_plain_answer_ends_the_session(tmp_path: Path) -> None:
    model = _ScriptedModel([_reply(content="Nothing to change.")])
    result = await ShelfAgent(model, "m", DiskCache(tmp_path)).run(PACK, "hi")  # type: ignore[arg-type]
    assert result.summary == "Nothing to change." and result.actions == []


async def test_run_agent_saves_pack_and_needs_a_goal(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    settings = Settings(_env_file=None, cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="goal"):
        await run_agent(settings, pack_dir, AgentRequest(), offline=True)
    with pytest.raises(OfflineCacheMissError):
        await run_agent(settings, pack_dir, AgentRequest(goal="sell fast"), offline=True)
    assert load_pack(pack_dir) == PACK


def test_web_agent_and_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_agent(
        _settings: Settings, pack_dir: Path, request: AgentRequest, **_: object
    ) -> AgentOutcome:
        pack = load_pack(pack_dir)
        lamp = pack.items[0]
        lamp.buyer_replies = [
            *lamp.buyer_replies,
            BuyerReply(buyer_message="Still there?", draft_reply="Yes!"),
        ]
        save_pack(pack_dir, pack)
        return AgentOutcome(request.goal, ["item-001: reply drafted"], "Drafted one reply.")

    monkeypatch.setattr(web, "run_agent", fake_run_agent)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs"
    )
    with TestClient(web.create_app(settings)) as client:
        assert client.post("/api/runs/demo/agent", data={"goal": "x"}).status_code == 404
        client.post("/api/demo")
        response = client.post("/api/runs/demo/agent", data={"goal": "Sell fast"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["summary"] == "Drafted one reply."
        replies = body["pack"]["items"][0]["buyer_replies"]
        index = len(replies) - 1
        approve = client.post(f"/api/runs/demo/items/item-001/replies/{index}/approve")
        assert approve.json() == {"approved": True}
        pack = client.get("/api/runs/demo").json()["pack"]
        assert pack["items"][0]["buyer_replies"][index]["approved"] is True
        assert client.post("/api/runs/demo/items/item-001/replies/99/approve").status_code == 404
