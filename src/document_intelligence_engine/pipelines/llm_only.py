"""LLM-only extraction baseline."""

from __future__ import annotations

import json
import time
from typing import Any

from ..domain.experiment_models import ExtractionOutput, ProcessedDocument
from ..llm.client import LLMClient
from ..llm.prompts import (
    DEFAULT_EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from .base import BasePipeline


EXPECTED_FIELDS = ("invoice_number", "date", "vendor", "total_amount", "line_items")


class LLMOnlyPipeline(BasePipeline):
    """Prompt the LLM with full OCR text and request schema-constrained JSON."""

    name = "llm_only"

    def __init__(self, config: dict[str, Any] | None = None, client: LLMClient | None = None) -> None:
        self.config = config or {}
        self.client = client or LLMClient(
            backend=str(self.config.get("backend", "nvidia")),
            model=str(self.config.get("model", "nv-mistralai/mistral-nemo-12b-instruct")),
            cache_dir=str(self.config.get("cache_dir", "experiments/cache/llm")),
            base_url=self.config.get("base_url"),
        )
        self.max_input_tokens = int(self.config.get("max_input_tokens", 4000))
        self.max_retries = int(self.config.get("max_retries", 2))

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        started_at = time.perf_counter()
        ocr_text = _truncate_text(str(document.get("ocr_text", "")), self.max_input_tokens)
        prompt = EXTRACTION_USER_TEMPLATE.format(
            schema=self.config.get("schema", DEFAULT_EXTRACTION_SCHEMA),
            ocr_text=ocr_text,
        )

        errors: list[str] = []
        total_cost = 0.0
        raw_response = ""
        extracted_fields: dict[str, Any] = {}
        self.client.reset_call_tracking()

        for attempt in range(self.max_retries + 1):
            try:
                payload = self.client.extract_json(
                    prompt,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    max_tokens=int(self.config.get("max_tokens", 512)),
                    temperature=float(self.config.get("temperature", 0.0)),
                )
                raw_response = str(payload["text"])
                total_cost += float(self.client.last_call_cost)
                extracted_fields = _coerce_output(payload["parsed"])
                break
            except Exception as exc:
                errors.append(f"attempt_{attempt + 1}: {exc}")
        else:
            extracted_fields = {field: None for field in EXPECTED_FIELDS}
            extracted_fields["line_items"] = []

        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        confidences = {field: 0.0 for field in extracted_fields}

        return {
            "extracted_fields": extracted_fields,
            "confidences": confidences,
            "_constraint_flags": [],
            "_errors": errors,
            "latency_ms": latency_ms,
            "cost_usd": round(total_cost, 6),
            "raw_response": raw_response,
        }


def _truncate_text(text: str, token_limit: int) -> str:
    try:
        import tiktoken
    except ImportError:
        words = text.split()
        if len(words) <= token_limit:
            return text
        return " ".join(words[:token_limit])

    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_ids = encoding.encode(text)
    if len(token_ids) <= token_limit:
        return text
    return encoding.decode(token_ids[:token_limit])


def _coerce_output(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, received: {type(parsed).__name__}")
    extracted = dict(parsed)
    for field in EXPECTED_FIELDS:
        if field not in extracted:
            extracted[field] = [] if field == "line_items" else None

    if extracted["line_items"] is None:
        extracted["line_items"] = []
    if not isinstance(extracted["line_items"], list):
        try:
            extracted["line_items"] = json.loads(str(extracted["line_items"]))
        except json.JSONDecodeError:
            extracted["line_items"] = []
    return extracted
