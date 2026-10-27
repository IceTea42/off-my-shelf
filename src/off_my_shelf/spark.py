"""Muse Spark structured output: scene inventory and per-item listing analysis.

Every call asks for JSON matching a Pydantic model, validates the reply, and retries once with
the validation error when the model drifts from the schema. Raw replies are cached by request.
"""

import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from off_my_shelf.cache import DiskCache, OfflineCacheMissError, cache_key
from off_my_shelf.costs import token_cost
from off_my_shelf.imaging import data_url, encode
from off_my_shelf.schema import ItemAnalysis, Language, ModelCall, PipelineStep

MAX_SCENE_ITEMS = 12
LANGUAGE_NAMES = {Language.EN: "English", Language.FR: "French", Language.TR: "Turkish"}


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


class SceneInventory(BaseModel):
    """Distinct items worth listing in one photo."""

    model_config = ConfigDict(extra="forbid")

    items: list[SceneItem] = Field(max_length=MAX_SCENE_ITEMS)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema with refs inlined, every property required and no extras.

    Structured-output endpoints are strictest about these three points; validation of the
    reply against the Pydantic model still enforces defaults and value constraints.
    """
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.pop("$defs", {})

    def walk(node: Any) -> Any:  # noqa: ANN401 - JSON Schema nodes are untyped
        if isinstance(node, list):
            return [walk(child) for child in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {k: walk(v) for k, v in node.items() if k not in {"title", "default"}}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    result: dict[str, Any] = walk(schema)
    return result


@dataclass(frozen=True, slots=True)
class SparkResult[T: BaseModel]:
    value: T
    call: ModelCall


ANALYZE_SYSTEM = """You are an expert second-hand reseller in France who writes listings for \
Vinted, Leboncoin and eBay. You look at one photo of one item and return its listing as JSON.
Rules:
- Identify the item precisely; give brand and model only when visible or certain, else null.
- Grade the condition from what you see; list visible defects plainly, never invent any.
- Price in EUR for the French second-hand market: low = sells within days, high = patient \
seller; suggested sits between them. Explain the price in one sentence.
- Write the title and description in {language}; be honest, specific and friendly; no emoji, \
no ALL CAPS, no contact details. Tags are short search words in {language}.
- Triage: sell if it is worth more than about 5 EUR and in working order; donate if usable but \
low value; recycle if broken or unsafe. Give the reason in one sentence."""

INVENTORY_SYSTEM = f"""You list the distinct physical items in a photo that someone could sell, \
donate or recycle. Ignore walls, floors, shelves and furniture that is clearly part of the room \
unless it is the obvious subject. Return at most {MAX_SCENE_ITEMS} items, largest first. Each \
phrase is a short English noun phrase a segmentation model can find, unique within the photo."""


class SparkAnalyzer:
    """Muse Spark calls; ``client=None`` means cache-only (offline)."""

    def __init__(
        self, client: AsyncOpenAI | None, model: str, cache: DiskCache, max_tokens: int
    ) -> None:
        self.client = client
        self.model = model
        self.cache = cache
        self.max_tokens = max_tokens

    async def inventory(self, image: Image.Image) -> SparkResult[SceneInventory]:
        content = [
            {"type": "text", "text": "List the items in this photo."},
            _image_part(image),
        ]
        return await self._structured(
            PipelineStep.ANALYZE, INVENTORY_SYSTEM, content, SceneInventory
        )

    async def analyze(
        self, crop: Image.Image, language: Language, note: str | None = None
    ) -> SparkResult[ItemAnalysis]:
        text = "Create the listing for this item."
        if note:
            text += f"\nThe owner says: {note}"
        system = ANALYZE_SYSTEM.format(language=LANGUAGE_NAMES[language])
        content = [{"type": "text", "text": text}, _image_part(crop)]
        result = await self._structured(PipelineStep.ANALYZE, system, content, ItemAnalysis)
        if result.value.listing.language is not language:
            listing = result.value.listing.model_copy(update={"language": language})
            fixed = result.value.model_copy(update={"listing": listing})
            return SparkResult(fixed, result.call)
        return result

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
        schema = strict_json_schema(schema_model)
        key = cache_key("spark", self.model, schema_model.__name__, schema, messages)
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            value = schema_model.model_validate_json(str(cached["content"]))
            return SparkResult(value, self._call(step, key, cached, latency_ms=None))
        client = self.client
        if client is None:
            raise OfflineCacheMissError(step.value, key)

        start = time.perf_counter()
        reply = await self._complete(client, messages, schema_model.__name__, schema)
        try:
            value = schema_model.model_validate_json(reply["content"])
        except ValidationError as error:
            repair = [
                *messages,
                {"role": "assistant", "content": reply["content"]},
                {"role": "user", "content": f"That JSON is invalid: {error}. Return fixed JSON."},
            ]
            retry = await self._complete(client, repair, schema_model.__name__, schema)
            reply = _merge_usage(reply, retry)
            try:
                value = schema_model.model_validate_json(reply["content"])
            except ValidationError as final_error:
                raise SparkReplyError(str(final_error)) from final_error
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
        response = await client.chat.completions.create(
            model=self.model,
            messages=wire_messages,
            response_format=response_format,
            max_tokens=self.max_tokens,
        )
        usage = response.usage
        return {
            "content": response.choices[0].message.content or "",
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        }

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


def _image_part(image: Image.Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url(encode(image, "JPEG"), "JPEG")}}


def _merge_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """The second reply's content with both calls' token counts summed."""
    merged = dict(second)
    for field in ("input_tokens", "output_tokens"):
        if first.get(field) is not None and second.get(field) is not None:
            merged[field] = first[field] + second[field]
    return merged
