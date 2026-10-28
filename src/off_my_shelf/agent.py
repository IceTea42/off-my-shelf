"""Shelf agent: Muse Spark with tools that act on a listing pack, from a typed or spoken goal.

"I'm moving next Friday, sell what you can fast and give the books away" becomes pricing
modes, triage changes and buyer-reply drafts. Every tool call is checked before it touches the
pack (known item, legal values, a buyer reply never below the floor price and never pushing
off-platform contact or payment), and the model sees each check's result, so it can correct
itself. Buyer replies are drafts until a person approves them.
"""

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.config import Settings
from off_my_shelf.costs import token_cost
from off_my_shelf.export import export_pack
from off_my_shelf.meta_client import make_client
from off_my_shelf.mocks import load_pack, save_pack
from off_my_shelf.schema import (
    Bundle,
    BuyerReply,
    Item,
    ListingPack,
    ModelCall,
    PipelineStep,
    PricingMode,
    Triage,
    TriageDecision,
)
from off_my_shelf.voice import Transcriber

MAX_STEPS = 12
BUNDLE_FLOOR_SHARE = 0.7  # a lot may sell below single-item floors, but not by more than 30 %
SELL_FAST_AT, MAX_VALUE_AT = 0.2, 0.85  # where the asking price sits in the low..high range
PRICE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:€|eur|euros?)|(?:€|eur)\s*(\d+(?:[.,]\d+)?)", re.I)
OFF_PLATFORM = re.compile(
    r"@|https?://|www\.|\+?\d[\d\s.-]{7,}\d|whatsapp|paypal|western union|virement|iban", re.I
)

AGENT_SYSTEM = f"""You are the Off My Shelf agent. You help one person clear out their things by \
acting on their listing pack with the tools provided. Follow their goal; when it says nothing \
about an item, leave that item alone. Keep prices fair for the French second-hand market.
When the goal is to clear things quickly, group low-value items that belong together (same room \
or use, each worth under about 10 EUR) into one lot with make_bundle; a lot is priced below the \
sum of its items' asking prices but never below {BUNDLE_FLOOR_SHARE:.0%} of their floors.
Rules for buyer replies: polite, short, in the listing's language; never accept or suggest a \
price below the item's floor; keep the conversation and payment on the marketplace (no phone \
numbers, emails, links or bank transfers). Tool results tell you whether an action was accepted; \
fix rejected actions. Call finish once, with a two-sentence summary of what you did."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_items",
            "description": "All items with id, title, condition, price range, triage and mode.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_pricing_mode",
            "description": "sell_fast prices near the low end, max_value near the high end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "mode": {"type": "string", "enum": [m.value for m in PricingMode]},
                    "reason": {"type": "string"},
                },
                "required": ["item_id", "mode", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_triage",
            "description": "Decide whether an item is sold, donated or recycled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "decision": {"type": "string", "enum": [d.value for d in TriageDecision]},
                    "reason": {"type": "string"},
                },
                "required": ["item_id", "decision", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_buyer_reply",
            "description": "Draft an answer to a buyer's message; a person approves it later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "buyer_message": {"type": "string"},
                    "reply": {"type": "string"},
                },
                "required": ["item_id", "buyer_message", "reply"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_bundle",
            "description": "Sell several low-value items together as one lot at one price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["item_ids", "title", "description", "price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the session with a short summary for the person.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


class AgentStepLimitError(RuntimeError):
    """The agent did not call finish within the step limit."""


@dataclass
class AgentResult:
    pack: ListingPack
    summary: str
    actions: list[str] = field(default_factory=list)
    calls: list[ModelCall] = field(default_factory=list)


def floor_price(item: Item) -> float:
    """The lowest price the agent may ever agree to for an item."""
    return item.analysis.price.low


def asking_price(item: Item, mode: PricingMode) -> float:
    price = item.analysis.price
    position = SELL_FAST_AT if mode is PricingMode.SELL_FAST else MAX_VALUE_AT
    return float(round(price.low + position * (price.high - price.low)))


def check_buyer_reply(item: Item, reply: str) -> str | None:
    """Why a draft reply is unacceptable, or None when it passes."""
    if OFF_PLATFORM.search(reply):
        return "keep contact and payment on the marketplace: no phone, email, link or transfer"
    floor = floor_price(item)
    for match in PRICE.finditer(reply):
        amount = float((match.group(1) or match.group(2)).replace(",", "."))
        if amount < floor:
            return f"{amount:g} is below this item's floor of {floor:g}"
    return None


def check_bundle(items: list[Item], price: float, already_bundled: set[str]) -> str | None:
    """Why a lot is unacceptable, or None when it passes."""
    if len(items) < 2:  # noqa: PLR2004 - a lot is at least two items
        return "a lot needs at least two different items"
    if already_bundled:
        return f"already in a lot: {', '.join(sorted(already_bundled))}"
    if any(i.analysis.triage.decision is TriageDecision.RECYCLE for i in items):
        return "items to recycle cannot be sold in a lot"
    if len({i.analysis.price.currency for i in items}) > 1:
        return "items are priced in different currencies"
    floor = BUNDLE_FLOOR_SHARE * sum(floor_price(i) for i in items)
    if price < floor:
        return f"{price:g} is below the lot's floor of {floor:.0f}"
    return None


class _Session:
    """Tool implementations over one pack; each returns the text the model sees."""

    def __init__(self, pack: ListingPack) -> None:
        self.items = {item.item_id: item.model_copy(deep=True) for item in pack.items}
        self.order = [item.item_id for item in pack.items]
        self.pack = pack
        self.bundles = [bundle.model_copy(deep=True) for bundle in pack.bundles]
        self.actions: list[str] = []
        self.summary: str | None = None

    def result(self) -> ListingPack:
        items = [self.items[i] for i in self.order]
        return self.pack.model_copy(update={"items": items, "bundles": self.bundles})

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handlers: dict[str, Callable[..., str]] = {
            "list_items": self.list_items,
            "set_pricing_mode": self.set_pricing_mode,
            "set_triage": self.set_triage,
            "draft_buyer_reply": self.draft_buyer_reply,
            "make_bundle": self.make_bundle,
            "finish": self.finish,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name}"
        try:
            return handler(**args)
        except (TypeError, ValueError, KeyError) as error:
            return f"error: {error}"

    def _item(self, item_id: str) -> Item:
        if item_id not in self.items:
            raise KeyError(f"unknown item_id {item_id!r}; call list_items")
        return self.items[item_id]

    def list_items(self) -> str:
        rows = []
        for item_id in self.order:
            a = self.items[item_id].analysis
            rows.append(
                {
                    "item_id": item_id,
                    "title": a.listing.title,
                    "language": a.listing.language.value,
                    "condition": a.condition.grade.value,
                    "price": {
                        "low": a.price.low,
                        "asking": a.price.suggested,
                        "high": a.price.high,
                    },
                    "currency": a.price.currency,
                    "triage": a.triage.decision.value,
                    "pricing_mode": self.items[item_id].pricing_mode,
                    "bundle": self._bundle_of(item_id),
                }
            )
        return json.dumps(rows, ensure_ascii=False)

    def set_pricing_mode(self, item_id: str, mode: str, reason: str) -> str:
        item = self._item(item_id)
        chosen = PricingMode(mode)
        asking = asking_price(item, chosen)
        price = item.analysis.price.model_copy(
            update={"suggested": asking, "rationale": f"{reason[:250]} ({chosen.value})"}
        )
        item.pricing_mode = chosen
        item.analysis = item.analysis.model_copy(update={"price": price})
        self.actions.append(f"{item_id}: {chosen.value} at {asking:g} {price.currency}")
        return f"ok: {item_id} now asks {asking:g} {price.currency}"

    def set_triage(self, item_id: str, decision: str, reason: str) -> str:
        item = self._item(item_id)
        triage = Triage(decision=TriageDecision(decision), reason=reason[:300] or decision)
        item.analysis = item.analysis.model_copy(update={"triage": triage})
        self.actions.append(f"{item_id}: {decision} ({triage.reason})")
        return f"ok: {item_id} set to {decision}"

    def draft_buyer_reply(self, item_id: str, buyer_message: str, reply: str) -> str:
        item = self._item(item_id)
        problem = check_buyer_reply(item, reply)
        if problem:
            self.actions.append(f"{item_id}: reply blocked by check ({problem}), rewriting")
            return f"rejected: {problem}. Write a new reply."
        draft = BuyerReply(buyer_message=buyer_message[:1000], draft_reply=reply[:1000])
        item.buyer_replies = [*item.buyer_replies, draft]
        self.actions.append(f"{item_id}: reply drafted, waiting for approval")
        return "ok: draft saved for human approval"

    def _bundle_of(self, item_id: str) -> str | None:
        return next((b.bundle_id for b in self.bundles if item_id in b.item_ids), None)

    def make_bundle(self, item_ids: list[str], title: str, description: str, price: float) -> str:
        items = [self._item(item_id) for item_id in dict.fromkeys(item_ids)]
        problem = check_bundle(
            items, price, {i.item_id for i in items if self._bundle_of(i.item_id)}
        )
        if problem:
            self.actions.append(f"lot blocked by check ({problem}), rewriting")
            return f"rejected: {problem}."
        bundle = Bundle(
            bundle_id=f"lot-{len(self.bundles) + 1:03d}",
            item_ids=[item.item_id for item in items],
            title=title[:80],
            description=description[:2000],
            price=float(round(price)),
            currency=items[0].analysis.price.currency,
        )
        self.bundles.append(bundle)
        self.actions.append(
            f"{bundle.bundle_id}: {len(items)} items as '{bundle.title}' "
            f"at {bundle.price:g} {bundle.currency}"
        )
        return f"ok: {bundle.bundle_id} created"

    def finish(self, summary: str) -> str:
        self.summary = summary.strip() or "Done."
        return "ok"


class ShelfAgent:
    """Tool-calling loop on Muse Spark; ``client=None`` replays a cached session offline."""

    def __init__(
        self,
        client: AsyncOpenAI | None,
        model: str,
        cache: DiskCache,
        reasoning_effort: str | None = "low",
    ) -> None:
        self.client = client
        self.model = model
        self.cache = cache
        self.reasoning_effort = reasoning_effort

    async def run(
        self, pack: ListingPack, goal: str, buyer_messages: dict[str, str] | None = None
    ) -> AgentResult:
        session = _Session(pack)
        request = goal.strip()
        for item_id, buyer_message in (buyer_messages or {}).items():
            request += f"\nBuyer message about {item_id}: {buyer_message}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": request},
        ]
        calls: list[ModelCall] = []
        for _ in range(MAX_STEPS):
            message, call = await self._step(messages)
            calls.append(call)
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                session.summary = session.summary or (message.get("content") or "").strip()
                break
            for tool_call in tool_calls:
                function = tool_call["function"]
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    output = "error: arguments are not valid JSON"
                else:
                    output = session.dispatch(function["name"], args)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": output}
                )
            if session.summary is not None:
                break
        else:
            raise AgentStepLimitError(f"no finish after {MAX_STEPS} steps")
        return AgentResult(session.result(), session.summary or "Done.", session.actions, calls)

    async def _step(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], ModelCall]:
        key = cache_key("agent", self.model, TOOLS, messages)
        cached = self.cache.get_json(key)
        latency_ms: int | None = None
        if isinstance(cached, dict):
            reply: dict[str, Any] = cached
        elif self.client is None:
            raise OfflineCacheMissError(PipelineStep.TRIAGE.value, key)
        else:
            start = time.perf_counter()
            wire_messages: Any = messages
            wire_tools: Any = TOOLS
            extra: dict[str, Any] = {}
            if self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort
            response = await self.client.chat.completions.create(
                model=self.model, messages=wire_messages, tools=wire_tools, **extra
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
            choice = response.choices[0].message
            usage = response.usage
            reply = {
                "message": {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (choice.tool_calls or [])
                    ]
                    or None,
                },
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            }
            self.cache.put_json(key, reply)
        tokens_in, tokens_out = reply.get("input_tokens"), reply.get("output_tokens")
        cost = None
        if tokens_in is not None and tokens_out is not None:
            cost = token_cost(self.model, tokens_in, tokens_out)
        call = ModelCall(
            step=PipelineStep.TRIAGE,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        message = {k: v for k, v in reply["message"].items() if v is not None}
        return message, call


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """What the person asks: a typed goal, a spoken one (WAV), and/or buyers to answer."""

    goal: str = ""
    voice: bytes | None = None
    buyer_messages: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    goal: str
    actions: list[str]
    summary: str


async def run_agent(
    settings: Settings, pack_dir: Path, request: AgentRequest, offline: bool = False
) -> AgentOutcome:
    """Run the shelf agent on a saved pack, then save and re-export it."""
    client = None if offline else make_client(settings)
    cache = DiskCache(settings.cache_dir)
    goal = request.goal.strip()
    if request.voice is not None:
        transcriber = Transcriber(client, settings.transcribe_model, cache)
        spoken, _ = await transcriber.transcribe(request.voice)
        goal = f"{goal}\n{spoken}".strip()
    if not goal and not request.buyer_messages:
        raise ValueError("give the agent a goal, a voice note or a buyer message")
    agent = ShelfAgent(client, settings.spark_model, cache, settings.reasoning_effort)
    result = await agent.run(load_pack(pack_dir), goal, request.buyer_messages)
    save_pack(pack_dir, result.pack)
    await asyncio.to_thread(export_pack, pack_dir)
    return AgentOutcome(goal, result.actions, result.summary)
