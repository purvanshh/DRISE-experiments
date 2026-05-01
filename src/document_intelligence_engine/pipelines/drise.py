"""Experiment wrapper for the existing deterministic DRISE pipeline."""

from __future__ import annotations

import time
from typing import Any

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.experiment_models import ExtractionOutput, ProcessedDocument
from document_intelligence_engine.pipelines.base import BasePipeline
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


class DRISEPipeline(BasePipeline):
    """Wrap the existing parser service behind the experiment interface."""

    name = "drise"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        parser_service: DocumentParserService | None = None,
    ) -> None:
        self.config = config or {}
        self._settings = get_settings()
        self._parser_service = parser_service or self._build_parser_service()
        self._local_cost_per_hour = float(self.config.get("local_cost_per_hour", 0.0))

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        image_path = document.get("image_path")
        if not image_path:
            raise ValueError("DRISEPipeline requires document['image_path'].")

        started_at = time.perf_counter()
        result = self._parser_service.parse_file(
            image_path,
            debug=bool(self.config.get("debug", False)),
            use_layout=bool(self.config.get("use_layout", True)),
            apply_constraints=bool(self.config.get("use_constraints", True)),
        )
        latency_ms = result["metadata"]["timing"].get("total") or round((time.perf_counter() - started_at) * 1000, 3)
        extracted_fields, confidences = _flatten_document(result["document"])
        cost_usd = (float(latency_ms) / 1000.0 / 3600.0) * self._local_cost_per_hour

        return {
            "extracted_fields": extracted_fields,
            "confidences": confidences,
            "_constraint_flags": list(result["document"].get("_constraint_flags", [])),
            "_errors": [str(error) for error in result["document"].get("_errors", [])],
            "latency_ms": float(latency_ms),
            "cost_usd": round(cost_usd, 6),
            "metadata": result["metadata"],
        }

    def _build_parser_service(self) -> DocumentParserService:
        settings = self._settings
        model_service = LayoutAwareModelService(settings)
        model_service.load()
        return DocumentParserService(settings, model_service)


def _flatten_document(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    extracted_fields: dict[str, Any] = {}
    confidences: dict[str, float] = {}
    for field_name, payload in document.items():
        if field_name.startswith("_"):
            continue
        if isinstance(payload, dict) and "value" in payload:
            extracted_fields[field_name] = payload.get("value")
            confidences[field_name] = round(float(payload.get("confidence", 0.0)), 6)
        else:
            extracted_fields[field_name] = payload
            confidences[field_name] = 0.0
    return extracted_fields, confidences
