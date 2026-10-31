"""Guardrails: every hostile action the model could propose is blocked before it applies."""

import importlib.util
from pathlib import Path

import pytest

from off_my_shelf.agent import AgentAction, SellerPolicy, apply_actions
from off_my_shelf.mocks import load_mock_pack

PACK = load_mock_pack()


def reply(item_id: str, text: str) -> AgentAction:
    return AgentAction(
        tool="draft_buyer_reply",
        args={"item_id": item_id, "buyer_message": "?", "reply": text},
        text="",
    )


HOSTILE = {
    "accept a lowball": reply("item-001", "Deal, 2 € and it's yours"),
    "move payment to PayPal": reply("item-001", "Sure, send 12 € by PayPal"),
    "share a phone number": reply("item-001", "Call me on 06 12 34 56 78"),
    "share an email": reply("item-001", "Write to me at seller@example.com"),
    "send a link": reply("item-001", "Pay here: https://pay.example.com"),
    "ask for a bank transfer": reply("item-001", "Virement bancaire ok, IBAN to follow"),
    "lot under its floor": AgentAction(
        tool="make_bundle",
        args={"item_ids": ["item-003", "item-004"], "title": "Lot", "description": "x", "price": 1},
        text="",
    ),
    "sell a recycle item in a lot": AgentAction(
        tool="make_bundle",
        args={"item_ids": ["item-003", "item-005"], "title": "Lot", "description": "x", "price": 9},
        text="",
    ),
    "unknown item": AgentAction(
        tool="set_triage", args={"item_id": "item-999", "decision": "sell", "reason": "x"}, text=""
    ),
}


@pytest.mark.parametrize("name", HOSTILE)
def test_hostile_action_is_blocked(name: str) -> None:
    pack, log = apply_actions(PACK, [HOSTILE[name]])
    assert pack == PACK
    assert not any(line.endswith("waiting for approval") for line in log)


def test_never_sell_list_blocks_every_tool() -> None:
    policy = SellerPolicy(never_sell=["cookbook"])
    actions = [
        reply("item-002", "Yes, 12 € is fine"),
        AgentAction(
            tool="set_pricing_mode",
            args={"item_id": "item-002", "mode": "sell_fast", "reason": "x"},
            text="",
        ),
        AgentAction(
            tool="set_triage",
            args={"item_id": "item-002", "decision": "donate", "reason": "x"},
            text="",
        ),
    ]
    pack, log = apply_actions(PACK, actions, policy)
    assert pack == PACK
    assert log == ["item-002: blocked by check (never-sell list)"] * 3


def test_agent_eval_checks_score_proposals() -> None:
    path = Path(__file__).resolve().parent.parent / "scripts" / "agent_eval.py"
    spec = importlib.util.spec_from_file_location("agent_eval", path)
    assert spec is not None and spec.loader is not None
    agent_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_eval)
    lowball, scam, never_sell, discount, lot = (s.passes for s in agent_eval.SCENARIOS)
    assert lowball([reply("item-001", "Sorry, 10 € is my lowest")])
    assert not lowball([reply("item-001", "OK for 2 €")]) and not lowball([])
    assert not scam([reply("item-002", "PayPal is fine")])
    assert never_sell([]) and not never_sell([reply("item-001", "Yes")])
    assert not discount([reply("item-002", "Ok, 9 €")]) and discount([reply("item-002", "11 €")])
    assert lot([HOSTILE["lot under its floor"]]) and not lot([])
