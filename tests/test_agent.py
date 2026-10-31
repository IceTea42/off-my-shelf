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
    AgentAction,
    AgentOutcome,
    AgentRequest,
    AgentStepLimitError,
    SellerPolicy,
    ShelfAgent,
    apply_actions,
    asking_price,
    check_bundle,
    check_buyer_reply,
    floor_price,
    run_agent,
)
from off_my_shelf.cache import DiskCache, OfflineCacheMissError
from off_my_shelf.config import Settings
from off_my_shelf.mocks import MOCK_PACK_DIR, load_mock_pack, load_pack
from off_my_shelf.schema import PricingMode, TriageDecision

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
    assert any("reply blocked by check (5 is below" in action for action in result.actions)
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


async def test_broken_tool_arguments_are_reported_back(tmp_path: Path) -> None:
    broken = SimpleNamespace(id="c1", function=SimpleNamespace(name="list_items", arguments="{"))
    model = _ScriptedModel([_reply(broken), _reply(content="Gave up.")])
    result = await ShelfAgent(model, "m", DiskCache(tmp_path)).run(PACK, "hi")  # type: ignore[arg-type]
    assert result.summary == "Gave up."
    assert model.sent[1][-1]["content"] == "error: arguments are not valid JSON"


async def test_run_agent_saves_pack_and_needs_a_goal(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    shutil.copytree(MOCK_PACK_DIR, pack_dir)
    settings = Settings(_env_file=None, cache_dir=tmp_path / "cache")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="goal"):
        await run_agent(settings, pack_dir, AgentRequest(), offline=True)
    with pytest.raises(OfflineCacheMissError):
        await run_agent(settings, pack_dir, AgentRequest(goal="sell fast"), offline=True)
    assert load_pack(pack_dir) == PACK


def test_web_agent_proposes_then_applies_selected_and_undoes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[AgentRequest] = []

    async def fake_run_agent(
        _settings: Settings, pack_dir: Path, request: AgentRequest, **kwargs: object
    ) -> AgentOutcome:
        assert kwargs == {"apply": False}
        seen.append(request)
        proposal = [
            AgentAction(
                tool="set_pricing_mode",
                args={"item_id": "item-001", "mode": "sell_fast", "reason": "moving"},
                text="item-001: sell_fast at 10 EUR",
            ),
            AgentAction(
                tool="draft_buyer_reply",
                args={"item_id": "item-001", "buyer_message": "Still there?", "reply": "Yes!"},
                text="item-001: reply drafted, waiting for approval",
            ),
        ]
        return AgentOutcome(request.goal, [a.text for a in proposal], "Two ideas.", proposal, 0.0)

    monkeypatch.setattr(web, "run_agent", fake_run_agent)
    monkeypatch.setenv("META_API_KEY", "test-key")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs"
    )
    with TestClient(web.create_app(settings)) as client:
        assert client.post("/api/runs/demo/agent", data={"goal": "x"}).status_code == 404
        client.post("/api/demo")
        form = {"goal": "Sell fast", "never_sell": "pokemon, item-005", "max_discount_pct": "20"}
        body = client.post("/api/runs/demo/agent", data=form).json()
        assert body["proposal"] == [
            "item-001: sell_fast at 10 EUR",
            "item-001: reply drafted, waiting for approval",
        ]
        assert seen[0].policy == SellerPolicy(never_sell=["pokemon", "item-005"], max_discount=0.2)
        before = client.get("/api/runs/demo").json()["pack"]
        assert before["items"][0]["analysis"]["price"]["suggested"] == 12  # nothing applied yet

        applied = client.post("/api/runs/demo/agent/apply", data={"selected": "1"}).json()
        lamp = applied["pack"]["items"][0]
        assert lamp["analysis"]["price"]["suggested"] == 12  # pricing was not picked
        assert lamp["buyer_replies"][-1]["draft_reply"] == "Yes!"
        assert applied["actions"] == ["item-001: reply drafted, waiting for approval"]
        assert client.post("/api/runs/demo/agent/apply").status_code == 404  # used up

        index = len(lamp["buyer_replies"]) - 1
        approve = client.post(f"/api/runs/demo/items/item-001/replies/{index}/approve")
        assert approve.json() == {"approved": True}
        assert client.post("/api/runs/demo/items/item-001/replies/99/approve").status_code == 404

        undone = client.post("/api/runs/demo/agent/undo").json()
        assert undone["pack"] == before and undone["can_undo"] is False
        assert client.post("/api/runs/demo/agent/undo").status_code == 404


def test_policy_blocks_never_sell_items_and_deep_discounts() -> None:
    policy = SellerPolicy(never_sell=["lampe"], max_discount=0.1)
    pack, log = apply_actions(
        PACK,
        [
            AgentAction(
                tool="set_triage",
                args={"item_id": "item-001", "decision": "donate", "reason": "x"},
                text="",
            ),
            AgentAction(
                tool="draft_buyer_reply",
                args={"item_id": "item-002", "buyer_message": "8?", "reply": "Ok for 9 €"},
                text="",
            ),
        ],
        policy,
    )
    assert pack.items[0] == PACK.items[0]  # "Lampe ..." is on the never-sell list
    assert log[0] == "item-001: blocked by check (never-sell list)"
    assert "9 is below this item's floor of 11" in log[1]  # 12 asking, at most 10 % off
    assert floor_price(PACK.items[1], policy) == 11 and floor_price(PACK.items[1]) == 6


def test_web_agent_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs"
    )
    with TestClient(web.create_app(settings)) as client:
        client.post("/api/demo")
        response = client.post("/api/runs/demo/agent", data={"goal": "sell fast"})
        assert response.status_code == 503
        assert "META_API_KEY" in response.json()["detail"]


def test_bundle_checks() -> None:
    third, fourth, recycle = PACK.items[2], PACK.items[3], PACK.items[4]  # floors 5 and 2
    assert check_bundle([third, fourth], 5, set()) is None  # 0.7 x 7 = 4.9
    assert "below the lot's floor" in (check_bundle([third, fourth], 4, set()) or "")
    assert "two different" in (check_bundle([third], 5, set()) or "")
    assert "recycle" in (check_bundle([third, recycle], 5, set()) or "")
    assert "already in a lot" in (check_bundle([third, fourth], 5, {"item-003"}) or "")


async def test_agent_makes_a_checked_lot(tmp_path: Path) -> None:
    ids = ["item-003", "item-004"]
    model = _ScriptedModel(
        [
            _reply(
                _tool(
                    "make_bundle",
                    "b1",
                    item_ids=ids,
                    title="Kitchen lot",
                    description="Two",
                    price=2,
                )
            ),
            _reply(
                _tool(
                    "make_bundle",
                    "b2",
                    item_ids=ids,
                    title="Kitchen lot",
                    description="Two",
                    price=6,
                )
            ),
            _reply(
                _tool("make_bundle", "b3", item_ids=ids, title="Again", description="Two", price=6)
            ),
            _reply(_tool("finish", "b4", summary="Made a kitchen lot.")),
        ]
    )
    agent = ShelfAgent(model, "muse-spark-1.3", DiskCache(tmp_path))  # type: ignore[arg-type]
    result = await agent.run(PACK, "clear the kitchen fast")

    assert [
        b.model_dump(include={"bundle_id", "item_ids", "price"}) for b in result.pack.bundles
    ] == [{"bundle_id": "lot-001", "item_ids": ids, "price": 6.0}]
    assert result.actions[0].startswith("lot blocked by check (2 is below")
    assert result.actions[1] == "lot-001: 2 items as 'Kitchen lot' at 6 EUR"
    assert "already in a lot" in result.actions[2]
