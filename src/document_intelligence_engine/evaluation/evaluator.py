"""Per-document evaluator for experiment outputs."""

from __future__ import annotations

from typing import Any

from document_intelligence_engine.domain.experiment_models import ExtractionOutput

from . import metrics


DEFAULT_SCHEMA = {
    "invoice_number": str,
    "date": str,
    "vendor": str,
    "total_amount": (int, float),
    "line_items": list,
}


class Evaluator:
    """Evaluate a single prediction against a document ground truth."""

    def __init__(
        self,
        *,
        fields: list[str] | None = None,
        schema: dict[str, type | tuple[type, ...]] | None = None,
    ) -> None:
        self.fields = fields or list(metrics.DEFAULT_FIELDS)
        self.schema = schema or DEFAULT_SCHEMA

    def evaluate(
        self,
        output: ExtractionOutput,
        ground_truth: dict[str, Any] | None,
        source_text: str = "",
    ) -> dict[str, Any]:
        prediction = dict(output.get("extracted_fields", {}))
        reference = ground_truth or {}
        field_scores = metrics.compute_field_f1_scores(prediction, reference, self.fields)
        return {
            "field_f1": field_scores,
            "field_level_f1": metrics.compute_field_level_f1(prediction, reference, self.fields),
            "exact_match": metrics.compute_document_exact_match(prediction, reference, self.fields),
            "schema_valid": metrics.compute_schema_validity(prediction, self.schema),
            "hallucination_rate": metrics.compute_hallucination_rate(prediction, source_text),
            "latency_ms": float(output.get("latency_ms", 0.0)),
            "cost_usd": float(output.get("cost_usd", 0.0)),
            "errors": list(output.get("_errors", [])),
            "constraint_flags": list(output.get("_constraint_flags", [])),
        }
