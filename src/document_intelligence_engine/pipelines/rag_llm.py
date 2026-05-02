"""RAG plus LLM extraction baseline."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from ..domain.experiment_models import ExtractionOutput, ProcessedDocument
from ..llm.client import LLMClient
from ..llm.client import _extract_json_candidate
from ..llm.prompts import EXTRACTION_SYSTEM_PROMPT, FIELD_EXTRACTION_TEMPLATE
from .base import BasePipeline
from ..retrieval.embedder import Embedder
from ..retrieval.retriever import DocumentRetriever


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
                prompt = FIELD_EXTRACTION_TEMPLATE.format(
                    field_name=field_name,
                    context="\n\n".join(retrieved_chunks),
                )
                payload = self.client.generate(
                    prompt,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    max_tokens=int(self.config.get("field_max_tokens", 256)),
                    temperature=float(self.config.get("temperature", 0.0)),
                )
                total_cost += _resolve_payload_cost(self.client, payload)
                raw_payloads[field_name] = str(payload["text"])
                extracted_fields[field_name] = _parse_field_response(field_name, payload["text"])
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


def _parse_field_response(field_name: str, raw_text: str) -> Any:
    parsed = _loads_with_repair(raw_text)
    if isinstance(parsed, dict) and field_name in parsed:
        parsed = parsed[field_name]
    if field_name == "line_items":
        return parsed if isinstance(parsed, list) else []
    return parsed


def _loads_with_repair(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(raw_text)
        if candidate is not None:
            return json.loads(candidate)

        stripped = raw_text.strip()
        if ":" in stripped:
            stripped = stripped.split(":", maxsplit=1)[1].strip()
        return json.loads(stripped)
