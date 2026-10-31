"""Muse Spark structured output: scene inventory and per-item listing analysis.

Live tests showed the endpoint does not always honour the JSON Schema it is given, so a reply
goes through four defences before it is accepted:
1. the schema sent on the wire keeps only types, enums and required keys (length and pattern
   limits are what make constrained decoding emit junk), and the prompt spells the shape out;
2. the reply is unwrapped from code fences, then trimmed to the real limits (long strings cut
   at a word, extra list items and unknown keys dropped, numbers clamped, junk tags removed);
3. if it is still invalid, the model gets one repair turn with the validation errors;
4. a reply that fails anyway is saved under ``<cache>/failures/`` for inspection.
"""

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import token_cost
from off_my_shelf.imaging import data_url, encode
from off_my_shelf.schema import ItemAnalysis, Language, ListingText, ModelCall, PipelineStep

MAX_SCENE_ITEMS = 12
LANGUAGE_NAMES = {Language.EN: "English", Language.FR: "French", Language.TR: "Turkish"}
CONSTRAINT_KEYS = frozenset(
    {"maxLength", "minLength", "maxItems", "minItems", "pattern", "minimum", "maximum"}
    | {"exclusiveMinimum", "exclusiveMaximum"}
)
ANNOTATION_KEYS = frozenset({"title", "default"})
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
JUNK_RUN = re.compile(r"(.)\1{3,}")  # a character repeated 4+ times: degenerate decoding


class SparkReplyError(RuntimeError):
    """Muse Spark did not return valid JSON for the schema, even after one repair attempt."""


class SceneItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(
        min_length=1,
        max_length=40,
        description="Short English noun phrase for a segmentation model, e.g. 'desk lamp'.",
    )
    label: str = Field(min_length=1, max_length=80, description="What the item is, briefly.")
    short_name: str = Field(
        min_length=1,
        max_length=24,
        description="1 or 2 lowercase English words naming the item, e.g. 'teapot', 'cake stand'.",
    )


class AdCopy(BaseModel):
    """The one line written on top of an item's video ad."""

    model_config = ConfigDict(extra="forbid")
    hook: str = Field(max_length=70)


class ShotCheck(BaseModel):
    """Is the AI product shot a faithful picture of the real item?"""

    model_config = ConfigDict(extra="forbid")

    same_item: bool
    extra_objects: bool
    added_defects: bool
    reason: str = Field(min_length=1, max_length=300)

    @property
    def passed(self) -> bool:
        return self.same_item and not self.extra_objects and not self.added_defects


class SceneInventory(BaseModel):
    """Distinct items worth listing in one photo."""

    model_config = ConfigDict(extra="forbid")

    items: list[SceneItem] = Field(max_length=MAX_SCENE_ITEMS)


# --- Schema shaping ------------------------------------------------------------------------


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema with refs inlined, every property required and no extras."""
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.pop("$defs", {})

    def walk(node: Any) -> Any:  # noqa: ANN401 - JSON Schema nodes are untyped
        if isinstance(node, list):
            return [walk(child) for child in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {k: walk(v) for k, v in node.items() if k not in ANNOTATION_KEYS}
        if "properties" in node:  # property names are data, never annotations
            out["properties"] = {name: walk(sub) for name, sub in node["properties"].items()}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    result: dict[str, Any] = walk(schema)
    return result


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema without value limits: what we ask the endpoint to decode against."""

    def walk(node: Any) -> Any:  # noqa: ANN401
        if isinstance(node, list):
            return [walk(child) for child in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items() if k not in CONSTRAINT_KEYS}
        if "properties" in node:  # property names are data, never constraints
            out["properties"] = {name: walk(sub) for name, sub in node["properties"].items()}
        return out

    result: dict[str, Any] = walk(schema)
    return result


def json_skeleton(schema: dict[str, Any]) -> str:
    """A one-line picture of the expected JSON, e.g. {"name": string, "tags": [string]}."""

    def shape(node: dict[str, Any]) -> str:
        if "anyOf" in node:
            return " | ".join(shape(option) for option in node["anyOf"])
        if "enum" in node:
            return " | ".join(json.dumps(value) for value in node["enum"])
        kind = node.get("type")
        if kind == "object":
            fields = ", ".join(f'"{k}": {shape(v)}' for k, v in node["properties"].items())
            return "{" + fields + "}"
        if kind == "array":
            return f"[{shape(node['items'])}]"
        return str(kind)

    return shape(schema)


def fit(value: Any, schema: dict[str, Any]) -> Any:  # noqa: ANN401, PLR0911
    """Bring a decoded reply within the schema's limits where that is safe to do."""
    if "anyOf" in schema:
        if value is None:
            return None
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        return fit(value, options[0]) if len(options) == 1 else value
    kind = schema.get("type")
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: fit(v, props[k]) for k, v in value.items() if k in props}
    if kind == "array" and isinstance(value, list):
        items = [fit(v, schema["items"]) for v in value]
        items = [v for v in items if not (isinstance(v, str) and JUNK_RUN.search(v))]
        return items[: schema["maxItems"]] if "maxItems" in schema else items
    if kind == "string" and isinstance(value, str):
        limit = schema.get("maxLength")
        if limit is not None and len(value) > limit:
            cut = value[:limit].rsplit(" ", 1)[0] if " " in value[:limit] else value[:limit]
            return cut.rstrip(" ,;:-")
        return value
    if kind in {"number", "integer"} and isinstance(value, int | float):
        low, high = schema.get("minimum"), schema.get("maximum")
        if low is not None and value < low:
            return low
        if high is not None and value > high:
            return high
    return value


def parse_reply[T: BaseModel](text: str, model: type[T], schema: dict[str, Any]) -> T:
    """Unwrap, trim and validate one reply; raises ValidationError or ValueError."""
    match = FENCE.match(text)
    data = json.loads(match.group(1) if match else text)
    return model.model_validate(fit(data, schema))


# --- Prompts ---------------------------------------------------------------------------------

ANALYZE_SYSTEM = """You are an expert second-hand reseller in France who writes listings for \
Vinted, Leboncoin and eBay. You look at one photo of one item and return its listing as JSON.
Rules:
- Identify the item precisely; give brand and model only when visible or certain, else null.
- Grade the condition from what you see; list visible defects plainly, never invent any.
- Price in EUR for the French second-hand market: low = sells within days, high = patient \
seller; suggested sits between them. Explain the price in one sentence.
- Write the title (at most 80 characters) and description (at most 1,000 characters) in \
{language}; be honest, specific and friendly; no emoji, no ALL CAPS, no contact details.
- Tags: up to 8 short search words in {language}.
- Triage: sell if it is worth more than about 5 EUR and in working order; donate if usable but \
low value; recycle if broken or unsafe but its materials can be recycled; bin if it is rubbish \
(packaging, worn out, dirty beyond use); keep if it is clearly not meant to go (the table or \
shelf the items stand on, something only partly in the photo). Give the reason in one sentence.
Reply with one JSON object only, no prose, in exactly this shape:
{skeleton}"""

INVENTORY_SYSTEM = """You list the distinct physical items in a photo that someone could sell, \
donate or recycle, including stands, tables, trays and other movable furniture. Ignore fixed \
parts of the room (walls, floors, windows, built-in shelves), food, and the surface the items \
lie on when the photo is a close-up of them. Return at most \
{limit} items, largest first. Each phrase is a short English noun phrase (2 to 4 words) a \
segmentation model can find, unique within the photo: tell look-alikes apart by colour, \
material or size rather than by position words.
Reply with one JSON object only, no prose, in exactly this shape:
{skeleton}"""


def correction_note(notes: str, previous_title: str) -> str:
    """The owner's corrections, as the note that goes with the crop when the listing is redone."""
    return (
        f'{notes}\nYour previous listing was titled "{previous_title}" and was wrong. The owner '
        "has the item in hand: trust them on what it is, its size, brand, material and "
        "condition, and rewrite the whole listing and price accordingly."
    )


AD_SYSTEM = """You write the hook line of a 5-second vertical video ad for one second-hand \
item, in {language}: at most 60 characters, catchy but honest (nothing that is not in the \
listing), no emoji, no hashtags, no price.
Reply with one JSON object only, no prose, in exactly this shape:
{skeleton}"""


TRANSLATE_SYSTEM = """You translate second-hand marketplace listings. Rewrite the listing in \
{language}: keep every fact, number, brand and model name and defect; use the wording a local \
seller would; title at most 80 characters; up to 8 tags in {language}.
Reply with one JSON object only, no prose, in exactly this shape:
{skeleton}"""


SHOT_CHECK_SYSTEM = """You check AI-completed product photos for a second-hand marketplace. \
Image 1 is the real item: {name}, with parts hidden behind other objects left blank. Image 2 is \
the same photo where only those hidden parts were completed. Answer:
- same_item: is image 2 the same kind of item, with the same shape, colours and materials?
- extra_objects: does image 2 show any object other than the item?
- added_defects: does image 2 show damage, dents, stains, dust or dirt that are not visible in \
image 1? Known real defects: {defects}.
- reason: one sentence explaining any problem, or "ok".
Reply with one JSON object only, no prose, in exactly this shape:
{skeleton}"""


@dataclass(frozen=True, slots=True)
class SparkResult[T: BaseModel]:
    value: T
    call: ModelCall


class SparkAnalyzer:
    """Muse Spark calls; ``client=None`` means cache-only (offline)."""

    def __init__(
        self,
        client: AsyncOpenAI | None,
        model: str,
        cache: DiskCache,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cache = cache
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    async def inventory(
        self, image: Image.Image, seen: Sequence[str] = ()
    ) -> SparkResult[SceneInventory]:
        """Items in one photo; ``seen`` are phrases from earlier frames of the same video, so
        the same item keeps the same phrase and its sightings can be merged."""
        text = "List the items in this photo."
        if seen:
            text += (
                " It is a later frame of a video. Items listed from earlier frames: "
                + "; ".join(seen)
                + ". When you see one of them again, reuse its exact phrase."
            )
        content = [
            {"type": "text", "text": text},
            _image_part(image),
        ]
        system = INVENTORY_SYSTEM.format(limit=MAX_SCENE_ITEMS, skeleton=_skeleton(SceneInventory))
        return await self._structured(PipelineStep.ANALYZE, system, content, SceneInventory)

    async def analyze(
        self,
        crop: Image.Image,
        language: Language,
        subject: str | None = None,
        note: str | None = None,
    ) -> SparkResult[ItemAnalysis]:
        """The listing for the ``subject`` (the inventory's phrase) in the crop: a crop around a
        sofa also shows the cushions on it, and the listing must be the sofa's."""
        text = "Create the listing for this item."
        if subject:
            text = (
                f"Create the listing for this item: {subject}. The photo is cropped around it; "
                "other things in view are not part of it."
            )
        if note:
            text += f"\nThe owner says: {note}"
        system = ANALYZE_SYSTEM.format(
            language=LANGUAGE_NAMES[language], skeleton=_skeleton(ItemAnalysis)
        )
        content = [{"type": "text", "text": text}, _image_part(crop)]
        result = await self._structured(PipelineStep.ANALYZE, system, content, ItemAnalysis)
        if result.value.listing.language is not language:
            listing = result.value.listing.model_copy(update={"language": language})
            fixed = result.value.model_copy(update={"listing": listing})
            return SparkResult(fixed, result.call)
        return result

    async def translate(self, listing: ListingText, language: Language) -> SparkResult[ListingText]:
        """The same listing in another language (text only, no image)."""
        system = TRANSLATE_SYSTEM.format(
            language=LANGUAGE_NAMES[language], skeleton=_skeleton(ListingText)
        )
        content = [{"type": "text", "text": listing.model_dump_json(exclude={"language"})}]
        result = await self._structured(PipelineStep.ANALYZE, system, content, ListingText)
        value = result.value.model_copy(update={"language": language})
        return SparkResult(value, result.call)

    async def ad_hook(self, listing: ListingText) -> SparkResult[AdCopy]:
        """A short, honest hook for the item's video ad, in the listing's language."""
        system = AD_SYSTEM.format(
            language=LANGUAGE_NAMES[listing.language], skeleton=_skeleton(AdCopy)
        )
        content = [
            {"type": "text", "text": listing.model_dump_json(include={"title", "description"})}
        ]
        return await self._structured(PipelineStep.ANALYZE, system, content, AdCopy)

    async def check_shot(
        self, real: Image.Image, shot: Image.Image, name: str, defects: list[str]
    ) -> SparkResult[ShotCheck]:
        system = SHOT_CHECK_SYSTEM.format(
            name=name,
            defects="; ".join(defects) if defects else "none",
            skeleton=_skeleton(ShotCheck),
        )
        content = [
            {"type": "text", "text": "Image 1 (real photo), then image 2 (generated):"},
            _image_part(real),
            _image_part(shot),
        ]
        return await self._structured(PipelineStep.PRODUCT_SHOT, system, content, ShotCheck)

    async def _structured[T: BaseModel](
        self,
        step: PipelineStep,
        system: str,
        content: list[dict[str, Any]],
        schema_model: type[T],
    ) -> SparkResult[T]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        strict = strict_json_schema(schema_model)
        wire = wire_schema(strict)
        key = cache_key(
            "spark", self.model, self.reasoning_effort, schema_model.__name__, wire, messages
        )
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            value = parse_reply(str(cached["content"]), schema_model, strict)
            return SparkResult(value, self._call(step, key, cached, latency_ms=None))
        client = self.client
        if client is None:
            raise OfflineCacheMissError(step.value, key)

        start = time.perf_counter()
        reply = await self._complete(client, messages, schema_model.__name__, wire)
        try:
            value = parse_reply(reply["content"], schema_model, strict)
        except (ValidationError, ValueError) as error:
            repair = [
                *messages,
                {"role": "assistant", "content": reply["content"]},
                {"role": "user", "content": _repair_prompt(error)},
            ]
            retry = await self._complete(client, repair, schema_model.__name__, wire)
            attempts = [reply, retry]
            reply = _merge_usage(reply, retry)
            try:
                value = parse_reply(reply["content"], schema_model, strict)
            except (ValidationError, ValueError) as final_error:
                saved = self._save_failure(key, attempts, final_error)
                raise SparkReplyError(f"{final_error} (replies saved to {saved})") from final_error
        latency_ms = round((time.perf_counter() - start) * 1000)
        self.cache.put_json(key, reply)
        return SparkResult(value, self._call(step, key, reply, latency_ms))

    async def _complete(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, Any]],
        name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        # Plain dicts on the wire; the SDK's TypedDicts are stricter than what we build here.
        wire_messages: Any = messages
        response_format: Any = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        }
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        response = await client.chat.completions.create(
            model=self.model,
            messages=wire_messages,
            response_format=response_format,
            max_tokens=self.max_tokens,
            **extra,
        )
        usage = response.usage
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        }

    def _save_failure(self, key: str, attempts: list[dict[str, Any]], error: Exception) -> str:
        failure_key = cache_key("failures", key, time.time())
        self.cache.put_json(
            failure_key, {"model": self.model, "error": str(error), "replies": attempts}
        )
        return str(self.cache.root / "failures")

    def _call(
        self, step: PipelineStep, key: str, reply: dict[str, Any], latency_ms: int | None
    ) -> ModelCall:
        tokens_in, tokens_out = reply.get("input_tokens"), reply.get("output_tokens")
        cost = None
        if tokens_in is not None and tokens_out is not None:
            cost = token_cost(self.model, tokens_in, tokens_out)
        return ModelCall(
            step=step,
            model=self.model,
            cache_key=key,
            cached=latency_ms is None,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
        )


def _skeleton(model: type[BaseModel]) -> str:
    return json_skeleton(wire_schema(strict_json_schema(model)))


def _repair_prompt(error: Exception) -> str:
    if isinstance(error, ValidationError):
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in error.errors()[:5]
        )
    else:
        problems = f"not valid JSON ({error})"
    return f"That reply is invalid: {problems}. Return the complete corrected JSON object only."


def _image_part(image: Image.Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url(encode(image, "JPEG"), "JPEG")}}


def _merge_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """The second reply's content with both calls' token counts summed."""
    merged = dict(second)
    for field in ("input_tokens", "output_tokens"):
        if first.get(field) is not None and second.get(field) is not None:
            merged[field] = first[field] + second[field]
    return merged


@dataclass(frozen=True, slots=True)
class ListingProvider:
    name: str  # "baseten", "modal" or "meta"
    analyzer: SparkAnalyzer


class ListingChain:
    """The listing step across providers: each one in order until one answers.

    Muse Spark Open on the partners comes first and Muse Spark on the Meta Model API last, so a
    partner outage or rate limit only slows the run down instead of failing it.
    """

    def __init__(self, providers: Sequence[ListingProvider]) -> None:
        if not providers:
            raise ValueError("at least one listing provider is needed")
        self.providers = list(providers)

    async def analyze(
        self, crop: Image.Image, language: Language, subject: str
    ) -> tuple[SparkResult[ItemAnalysis], list[str]]:
        """The listing, and the providers that failed before it (for the run log)."""
        failed: list[str] = []
        for provider in self.providers:
            try:
                result = await provider.analyzer.analyze(crop, language, subject)
            except (OpenAIError, SparkReplyError, OfflineCacheMissError) as error:
                failed.append(f"{provider.name} ({type(error).__name__})")
                if provider is self.providers[-1]:
                    raise
                continue
            label = None if provider.name == "meta" else provider.name
            return SparkResult(
                result.value, result.call.model_copy(update={"provider": label})
            ), failed
        raise AssertionError("unreachable: the last provider re-raises")
