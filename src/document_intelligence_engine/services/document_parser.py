"""Unified end-to-end document parsing service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from document_intelligence_engine.core.config import AppSettings
from document_intelligence_engine.core.logging import get_logger
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from ingestion.pipeline import process_document_with_metadata
from postprocessing.pipeline import postprocess_predictions


logger = get_logger(__name__)


class DocumentParserService:
    """Shared orchestration service for API, CLI, and scripts."""

    def __init__(self, settings: AppSettings, model_service: LayoutAwareModelService) -> None:
        self._settings = settings
        self._model_service = model_service

    @property
    def model_service(self) -> LayoutAwareModelService:
        return self._model_service

    def parse_file(self, file_path: str | Path, debug: bool = False) -> dict[str, Any]:
        started_at = time.perf_counter()
        ocr_result = process_document_with_metadata(str(file_path), debug=debug)
        model_started_at = time.perf_counter()
        raw_predictions = self._model_service.predict(ocr_result["ocr_tokens"])
        model_duration_ms = _elapsed_ms(model_started_at)

        postprocessing_started_at = time.perf_counter()
        structured_document = postprocess_predictions(raw_predictions)
        postprocessing_duration_ms = _elapsed_ms(postprocessing_started_at)

        timing = dict(ocr_result["timing"])
        timing["model"] = model_duration_ms
        timing["postprocessing"] = postprocessing_duration_ms
        timing["total"] = round((time.perf_counter() - started_at) * 1000, 3)

        metadata = {
            "filename": Path(file_path).name,
            "page_count": ocr_result["page_count"],
            "ocr_token_count": len(ocr_result["ocr_tokens"]),
            "confidence_summary": build_confidence_summary(structured_document),
            "timing": timing,
            "warnings": derive_warnings(
                ocr_tokens=ocr_result["ocr_tokens"],
                document=structured_document,
                page_count=ocr_result["page_count"],
            ),
            "model": {
                "name": self._model_service.name,
                "version": self._model_service.version,
                "device": self._model_service.device,
            },
        }

        result = {
            "document": structured_document,
            "metadata": metadata,
        }
        if debug:
            result["debug"] = {
                "ocr_tokens": ocr_result["ocr_tokens"][: self._settings.performance.max_debug_tokens],
                "page_stats": ocr_result.get("page_stats", []),
                "raw_predictions": raw_predictions[: self._settings.performance.max_debug_tokens],
                "intermediate": ocr_result.get("debug", {}),
            }
        logger.info(
            "document_parse_completed",
            extra={
                "file_path": str(file_path),
                "page_count": metadata["page_count"],
                "ocr_token_count": metadata["ocr_token_count"],
                "timing": timing,
            },
        )
        return result


def build_confidence_summary(document: dict[str, Any]) -> dict[str, Any]:
    confidences = [
        float(payload.get("confidence", 0.0))
        for field_name, payload in document.items()
        if not str(field_name).startswith("_") and isinstance(payload, dict) and "confidence" in payload
    ]
    dropped_fields = sum(
        1
        for error in document.get("_errors", [])
        if isinstance(error, dict) and error.get("code") in {"low_confidence", "ablation_dropped_low_confidence"}
    )
    if not confidences:
        return {"average": 0.0, "minimum": 0.0, "maximum": 0.0, "kept_fields": 0, "dropped_fields": dropped_fields}
    return {
        "average": round(sum(confidences) / len(confidences), 6),
        "minimum": round(min(confidences), 6),
        "maximum": round(max(confidences), 6),
        "kept_fields": len(confidences),
        "dropped_fields": dropped_fields,
    }


def derive_warnings(
    ocr_tokens: list[dict[str, Any]],
    document: dict[str, Any],
    page_count: int,
) -> list[str]:
    warnings: list[str] = []
    if not ocr_tokens:
        warnings.append("empty_document")
        return warnings

    average_confidence = sum(float(token.get("confidence", 0.0)) for token in ocr_tokens) / len(ocr_tokens)
    if len(ocr_tokens) < 3 or average_confidence < 0.55:
        warnings.append("noisy_ocr_output")

    if any(
        isinstance(error, dict) and error.get("code") == "missing_required_field"
        for error in document.get("_errors", [])
    ):
        warnings.append("missing_required_fields")

    if page_count > 1 and any(
        isinstance(error, dict) and error.get("code") == "conflicting_values"
        for error in document.get("_errors", [])
    ):
        warnings.append("multi_page_inconsistency")

    return warnings


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
