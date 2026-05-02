"""RAG plus LLM extraction baseline."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from ..llm.client import LLMClient
from ..llm.client import _extract_json_candidate
from ..llm.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FIELD_EXTRACTION_TEMPLATE,
    STRICT_EXTRACTION_SYSTEM_PROMPT,
    STRICT_FIELD_EXTRACTION_TEMPLATE,
)
from ..domain.experiment_models import ExtractionOutput, ProcessedDocument
from ..retrieval.embedder import Embedder
from ..retrieval.retriever import DocumentRetriever
from .base import BasePipeline


FIELD_QUERIES = {
    "invoice_number": "invoice number identifier",
    "date": "invoice or receipt date",
    "vendor": "vendor or merchant name",
    "total_amount": "total amount due",
    "line_items": "line items with description quantity and price",
}


class RAGLLMPipeline(BasePipeline):
    """Retrieve per-field context chunks, then ask the LLM for each field."""

    name = "rag_llm"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        client: LLMClient | None = None,
        retriever: DocumentRetriever | None = None,
    ) -> None:
        self.config = config or {}
        self.target_fields = tuple(self.config.get("target_fields", FIELD_QUERIES.keys()))
        self.client = client or LLMClient(
            backend=str(self.config.get("backend", "nvidia")),
            model=str(self.config.get("model", "meta/llama-3.2-1b-instruct")),
            cache_dir=str(self.config.get("cache_dir", "experiments/cache/llm")),
            base_url=self.config.get("base_url"),
        )
        self.retriever = retriever or DocumentRetriever(
            embedder=Embedder(str(self.config.get("embedder", "all-MiniLM-L6-v2"))),
            top_k=int(self.config.get("top_k", 3)),
            chunk_size=int(self.config.get("chunk_size", 500)),
            cache_dir=str(self.config.get("retrieval_cache_dir", "experiments/cache/retrieval")),
        )
        self.system_prompt = str(
            self.config.get(
                "system_prompt",
                STRICT_EXTRACTION_SYSTEM_PROMPT
                if self.config.get("prompt_variant", "strict_v1") == "strict_v1"
                else EXTRACTION_SYSTEM_PROMPT,
            )
        )
        self.field_prompt_template = str(
            self.config.get(
                "field_prompt_template",
                STRICT_FIELD_EXTRACTION_TEMPLATE
                if self.config.get("prompt_variant", "strict_v1") == "strict_v1"
                else FIELD_EXTRACTION_TEMPLATE,
            )
        )

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        started_at = time.perf_counter()
        ocr_text = str(document.get("ocr_text", ""))
        doc_id = str(document.get("doc_id") or _derive_doc_id(ocr_text))
        chunk_count = len(self.retriever.ensure_index(doc_id, ocr_text))

        extracted_fields: dict[str, Any] = {}
        errors: list[str] = []
        total_cost = 0.0
        raw_payloads: dict[str, str] = {}
        retrieved_contexts: dict[str, list[str]] = {}
        _reset_client_tracking(self.client)

        for field_name in self.target_fields:
            query = FIELD_QUERIES.get(field_name, field_name.replace("_", " "))
            try:
                retrieved_chunks = self.retriever.retrieve(query, doc_id, ocr_text=ocr_text)
                retrieved_contexts[field_name] = retrieved_chunks
                prompt = self.field_prompt_template.format(
                    field_name=field_name,
                    context="\n\n".join(retrieved_chunks),
                )
                payload = self.client.generate(
                    prompt,
                    system_prompt=self.system_prompt,
                    max_tokens=int(self.config.get("field_max_tokens", 256)),
                    temperature=float(self.config.get("temperature", 0.0)),
                )
                total_cost += _resolve_payload_cost(self.client, payload)
                raw_payloads[field_name] = str(payload["text"])
                extracted_fields[field_name] = _parse_field_response(
                    field_name,
                    payload["text"],
                    context="\n".join(retrieved_chunks),
                )
            except Exception as exc:
                errors.append(f"{field_name}: {exc}")
                extracted_fields[field_name] = [] if field_name == "line_items" else None

        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        confidences = {field: 0.0 for field in extracted_fields}

        return {
            "extracted_fields": extracted_fields,
            "confidences": confidences,
            "_constraint_flags": [],
            "_errors": errors,
            "latency_ms": latency_ms,
            "cost_usd": round(total_cost, 6),
            "metadata": {
                "chunk_count": chunk_count,
                "top_k": self.retriever.top_k,
                "raw_field_responses": raw_payloads,
                "retrieved_contexts": retrieved_contexts,
            },
        }


def _derive_doc_id(ocr_text: str) -> str:
    return hashlib.sha256(ocr_text.encode("utf-8")).hexdigest()[:16]


def _reset_client_tracking(client: Any) -> None:
    reset = getattr(client, "reset_call_tracking", None)
    if callable(reset):
        reset()


def _resolve_payload_cost(client: Any, payload: dict[str, Any]) -> float:
    if hasattr(client, "last_call_cost"):
        return float(getattr(client, "last_call_cost", 0.0) or 0.0)
    return float(payload.get("cost_usd", 0.0) or 0.0)


def _parse_field_response(field_name: str, raw_text: str, *, context: str = "") -> Any:
    return _parse_field_response_with_context(field_name, raw_text, context=context)


def _loads_with_repair(raw_text: str) -> Any:
    parsed = _try_load_json(raw_text)
    if parsed is not _UNPARSEABLE:
        return parsed

    candidate = _extract_json_candidate(raw_text)
    if candidate is not None:
        parsed = _try_load_json(candidate)
        if parsed is not _UNPARSEABLE:
            return parsed

    stripped = raw_text.strip()
    if ":" in stripped:
        parsed = _try_load_json(stripped.split(":", maxsplit=1)[1].strip())
        if parsed is not _UNPARSEABLE:
            return parsed

    for line in reversed([line.strip() for line in raw_text.splitlines() if line.strip()]):
        parsed = _try_load_json(line)
        if parsed is not _UNPARSEABLE:
            return parsed
        if ":" in line:
            parsed = _try_load_json(line.split(":", maxsplit=1)[1].strip())
            if parsed is not _UNPARSEABLE:
                return parsed

    raise json.JSONDecodeError("Could not parse field response as JSON.", raw_text, 0)


_UNPARSEABLE = object()


def _parse_field_response_with_context(field_name: str, raw_text: str, *, context: str) -> Any:
    try:
        parsed = _loads_with_repair(raw_text)
    except json.JSONDecodeError:
        parsed = _UNPARSEABLE

    if parsed is not _UNPARSEABLE:
        if isinstance(parsed, dict) and field_name in parsed:
            parsed = parsed[field_name]
        parsed = _coerce_field_value(field_name, parsed)
        if _has_meaningful_value(field_name, parsed):
            return parsed

    fallback = _regex_fallback(field_name, raw_text, context)
    if _has_meaningful_value(field_name, fallback):
        return fallback
    return [] if field_name == "line_items" else None


def _coerce_field_value(field_name: str, value: Any) -> Any:
    if field_name == "line_items":
        if isinstance(value, list):
            normalized_items = [_normalize_line_item(item) for item in value]
            return [item for item in normalized_items if item]
        return []
    if field_name == "total_amount":
        amount = _parse_amount(value)
        return amount if amount is not None else None
    if field_name == "date" and value is not None:
        text = _clean_scalar_text(value)
        return text or None
    if value is None:
        return None
    text = _clean_scalar_text(value)
    return text if text is not None else None


def _has_meaningful_value(field_name: str, value: Any) -> bool:
    if field_name == "line_items":
        return isinstance(value, list) and len(value) > 0
    return value is not None


def _try_load_json(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return _UNPARSEABLE


def _regex_fallback(field_name: str, raw_text: str, context: str) -> Any:
    combined = "\n".join(part for part in (raw_text, context) if part).strip()
    if not combined:
        return [] if field_name == "line_items" else None

    if field_name == "total_amount":
        total_match = re.search(
            r"\bTOTAL\b[^0-9-]*(-?\d[\d,]*(?:\.\d+)?)",
            combined,
            flags=re.IGNORECASE,
        )
        if total_match:
            return _parse_amount(total_match.group(1))
        amounts = re.findall(r"-?\d[\d,]*(?:\.\d+)?", combined)
        return _parse_amount(amounts[-1]) if amounts else None

    if field_name == "date":
        match = re.search(
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            combined,
        )
        return match.group(0) if match else None

    if field_name == "invoice_number":
        match = re.search(
            r"(?:invoice(?:\s+(?:number|no))?|receipt)\s*#?\s*[:\-]?\s*([A-Za-z0-9/_-]+)",
            combined,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else _fallback_scalar_text(raw_text)

    if field_name == "vendor":
        match = re.search(r"\bvendor\b\s*[:\-]?\s*(.+)", combined, flags=re.IGNORECASE)
        if match:
            return _clean_scalar_text(match.group(1))
        return _fallback_scalar_text(raw_text)

    if field_name == "line_items":
        return _extract_line_items_from_context(context or combined)

    return _fallback_scalar_text(raw_text)


def _fallback_scalar_text(raw_text: str) -> str | None:
    for line in reversed([line.strip() for line in raw_text.splitlines() if line.strip()]):
        if line.lower() == "null":
            continue
        candidate = line.split(":", maxsplit=1)[1].strip() if ":" in line else line
        cleaned = _clean_scalar_text(candidate)
        if cleaned:
            return cleaned
    return None


def _clean_scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'").strip()
    return text or None


def _parse_amount(value: Any) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]+", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_line_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    description = _clean_scalar_text(item.get("description"))
    quantity = _parse_amount(item.get("quantity"))
    unit_price = _parse_amount(item.get("unit_price"))
    if description is None and quantity is None and unit_price is None:
        return None
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": unit_price,
    }


def _extract_line_items_from_context(context: str) -> list[dict[str, Any]]:
    tokens = [token for token in re.split(r"\s+", context.strip()) if token]
    if not tokens:
        return []

    segment_tokens = _slice_line_item_segment(tokens)
    if not segment_tokens:
        return []

    items: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(segment_tokens):
        quantity = None
        if _is_number_token(segment_tokens[cursor]) and cursor + 2 < len(segment_tokens):
            quantity = _parse_amount(segment_tokens[cursor])
            cursor += 1

        description_tokens: list[str] = []
        while cursor < len(segment_tokens) and not _is_number_token(segment_tokens[cursor]):
            description_tokens.append(segment_tokens[cursor])
            cursor += 1

        if not description_tokens or cursor >= len(segment_tokens):
            break

        description = _clean_scalar_text(" ".join(description_tokens))
        unit_price = _parse_amount(segment_tokens[cursor])
        cursor += 1

        if description and description.upper() not in _LINE_ITEM_STOPWORDS:
            items.append(
                {
                    "description": description,
                    "quantity": quantity,
                    "unit_price": unit_price,
                }
            )

    return [item for item in (_normalize_line_item(item) for item in items) if item]


_LINE_ITEM_STOPWORDS = {"TOTAL", "CASH", "CHANGE", "CHANGED"}


def _slice_line_item_segment(tokens: list[str]) -> list[str]:
    if _looks_like_ledger_prefix(tokens):
        cursor = 0
        while cursor + 1 < len(tokens) and tokens[cursor].upper() in _LINE_ITEM_STOPWORDS and _is_number_token(tokens[cursor + 1]):
            cursor += 2
        return tokens[cursor:]

    for index, token in enumerate(tokens):
        if token.upper() in _LINE_ITEM_STOPWORDS:
            return tokens[:index]
    return tokens


def _looks_like_ledger_prefix(tokens: list[str]) -> bool:
    return (
        len(tokens) >= 2
        and tokens[0].upper() in _LINE_ITEM_STOPWORDS
        and _is_number_token(tokens[1])
    )


def _is_number_token(token: str) -> bool:
    return _parse_amount(token) is not None
