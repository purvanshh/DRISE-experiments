"""Cross-field consistency checks and deterministic corrections."""

from __future__ import annotations

from datetime import date
from math import isclose
from statistics import fmean
from typing import Any

from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)


def apply_constraints(
    document: dict[str, dict[str, Any]],
    settings: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], list[str]]:
    constrained = {field: dict(payload) for field, payload in document.items()}
    errors: list[dict[str, str]] = []
    flags: list[str] = []

    required_fields = set(settings.postprocessing.constraints.required_fields)
    for field_name in required_fields:
        if field_name not in constrained:
            constrained[field_name] = {"value": None, "confidence": 0.0, "valid": False}
            flags.append(f"missing_required_field:{field_name}")
            errors.append(
                _error(field_name, "missing_required_field", "Required field is missing from output.")
            )
        elif constrained[field_name].get("value") in (None, ""):
            constrained[field_name]["valid"] = False
            flags.append(f"missing_required_field:{field_name}")
            errors.append(
                _error(field_name, "missing_required_field", "Required field has no value.")
            )

    if not settings.postprocessing.constraints.allow_future_dates:
        today = date.today()
        for field_name in settings.postprocessing.normalization.date_fields:
            record = constrained.get(field_name)
            if not record or not record.get("value"):
                continue
            try:
                parsed = date.fromisoformat(str(record["value"]))
            except ValueError:
                continue
            if parsed > today:
                record["valid"] = False
                flags.append(f"future_date:{field_name}")
                errors.append(
                    _error(field_name, "future_date", "Date field contains a future date.")
                )

    if "line_items" in constrained:
        constrained, amount_errors, amount_flags = _enforce_amount_consistency(
            constrained,
            settings.postprocessing.constraints.amount_tolerance,
        )
        errors.extend(amount_errors)
        flags.extend(amount_flags)

    return constrained, errors, flags


def _enforce_amount_consistency(
    document: dict[str, dict[str, Any]],
    tolerance: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], list[str]]:
    errors: list[dict[str, str]] = []
    flags: list[str] = []
    line_items_record = document.get("line_items")
    if not line_items_record:
        return document, errors, flags

    line_items = line_items_record.get("value")
    if not isinstance(line_items, list):
        return document, errors, flags

    computed_total = 0.0
    line_item_confidences: list[float] = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        price = item.get("price")
        quantity = item.get("quantity", 1)
        if isinstance(price, (int, float)) and isinstance(quantity, (int, float)):
            computed_total += float(price) * float(quantity)
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            line_item_confidences.append(float(confidence))

    if "total_amount" not in document or document["total_amount"].get("value") in (None, ""):
        document["total_amount"] = {
            "value": round(computed_total, 2),
            "confidence": round(fmean(line_item_confidences), 6) if line_item_confidences else 0.0,
            "valid": True,
            "corrected": True,
        }
        logger.info("corrected_total_amount", extra={"value": computed_total})
        flags.append("corrected_total_amount_from_line_items")
        return document, errors, flags

    total_amount_record = document["total_amount"]
    total_amount = total_amount_record.get("value")
    if isinstance(total_amount, (int, float)) and not isclose(
        float(total_amount),
        computed_total,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        total_amount_record["valid"] = False
        flags.append("line_items_total_mismatch")
        errors.append(
            _error(
                "total_amount",
                "line_items_total_mismatch",
                "Line item sum does not match total amount.",
            )
        )

    return document, errors, flags


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
