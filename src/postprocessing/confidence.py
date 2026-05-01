"""Field confidence aggregation and threshold filtering."""

from __future__ import annotations

from typing import Any

from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)


def apply_confidence_policy(
    document: dict[str, dict[str, Any]],
    settings: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    threshold = settings.postprocessing.confidence.min_field_confidence
    drop_below_threshold = settings.postprocessing.confidence.drop_below_threshold
    retain_threshold = max(0.35, round(threshold - 0.2, 6))

    filtered_document: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    for field_name, record in document.items():
        confidence = round(float(record.get("confidence", 0.0)), 6)
        record["confidence"] = confidence

        if confidence < threshold:
            errors.append(
                _error(
                    field_name,
                    "low_confidence",
                    f"Field confidence {confidence:.6f} is below threshold {threshold:.6f}.",
                )
            )
            logger.info(
                "dropped_field",
                extra={"field": field_name, "confidence": confidence, "threshold": threshold},
            )
            if drop_below_threshold and confidence < retain_threshold:
                continue
            record["valid"] = False
            record["retained_low_confidence"] = True

        filtered_document[field_name] = record

    return filtered_document, errors


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
