"""RAG plus LLM extraction baseline."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from document_intelligence_engine.domain.experiment_models import ExtractionOutput, ProcessedDocument
from document_intelligence_engine.llm.client import LLMClient
from document_intelligence_engine.llm.prompts import FIELD_EXTRACTION_TEMPLATE
from document_intelligence_engine.pipelines.base import BasePipeline
from document_intelligence_engine.retrieval.embedder import Embedder
from document_intelligence_engine.retrieval.retriever import DocumentRetriever


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
        self.client = client or LLMClient(
            backend=str(self.config.get("backend", "mock")),
            model=str(self.config.get("model", "mock-llm")),
            cache_dir=str(self.config.get("cache_dir", "experiments/cache/llm")),
            base_url=self.config.get("base_url"),
        )
        self.retriever = retriever or DocumentRetriever(
            embedder=Embedder(str(self.config.get("embedder", "hashing"))),
            top_k=int(self.config.get("top_k", 3)),
            chunk_size=int(self.config.get("chunk_size", 500)),
            cache_dir=str(self.config.get("retrieval_cache_dir", "experiments/cache/retrieval")),
        )

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        started_at = time.perf_counter()
        ocr_text = str(document.get("ocr_text", ""))
        doc_id = str(document.get("doc_id") or _derive_doc_id(ocr_text))
        self.retriever.ensure_index(doc_id, ocr_text)

        extracted_fields: dict[str, Any] = {}
        errors: list[str] = []
        total_cost = 0.0
        raw_payloads: dict[str, str] = {}

        for field_name, query in FIELD_QUERIES.items():
            try:
                retrieved_chunks = self.retriever.retrieve(query, doc_id, ocr_text=ocr_text)
                prompt = FIELD_EXTRACTION_TEMPLATE.format(
                    field_name=field_name,
                    context="\n\n".join(retrieved_chunks),
                )
                payload = self.client.generate(
                    prompt,
                    max_tokens=int(self.config.get("field_max_tokens", 256)),
                    temperature=float(self.config.get("temperature", 0.0)),
                )
                total_cost += float(payload["cost_usd"])
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
            "metadata": {"raw_field_responses": raw_payloads},
        }


def _derive_doc_id(ocr_text: str) -> str:
    return hashlib.sha256(ocr_text.encode("utf-8")).hexdigest()[:16]


def _parse_field_response(field_name: str, raw_text: str) -> Any:
    parsed = json.loads(raw_text)
    if field_name == "line_items":
        return parsed if isinstance(parsed, list) else []
    return parsed
