"""Regex and semantic field validation."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)


def validate_fields(
    entities: list[dict[str, Any]],
    settings: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    document: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    regex_rules = settings.postprocessing.validation.regex_rules
    date_fields = set(settings.postprocessing.normalization.date_fields)
    currency_fields = set(settings.postprocessing.normalization.currency_fields)

    for entity in entities:
        field_name = entity["field"]
        value = entity.get("value")
        confidence = round(float(entity.get("confidence", 0.0)), 6)

        record = {
            "value": value,
            "confidence": confidence,
            "valid": True,
            "source_key": entity.get("key", field_name),
        }

        if field_name in document:
            existing = document[field_name]
            if existing["value"] != value:
                preferred = _preferred_record(existing, record, field_name, date_fields, currency_fields)
                document[field_name] = preferred
                errors.append(
                    _error(
                        field=field_name,
                        code="conflicting_values",
                        message="Conflicting values detected; highest-confidence value retained.",
                    )
                )
                logger.warning(
                    "conflicting_values",
                    extra={"field": field_name, "kept": preferred["value"]},
                )
            else:
                existing["confidence"] = round((existing["confidence"] + confidence) / 2.0, 6)
            continue

        document[field_name] = record

    for field_name, record in document.items():
        field_errors: list[dict[str, str]] = []
        value = record["value"]
        if value in (None, ""):
            record["valid"] = False
            field_errors.append(
                _error(field=field_name, code="missing_value", message="Field value is missing.")
            )
        else:
            regex_rule = regex_rules.get(field_name)
            if regex_rule and not re.fullmatch(regex_rule, str(value)):
                record["valid"] = False
                field_errors.append(
                    _error(
                        field=field_name,
                        code="regex_validation_failed",
                        message=f"Value '{value}' does not match the configured pattern.",
                    )
                )

            if field_name in date_fields and not _is_iso_date(str(value)):
                record["valid"] = False
                field_errors.append(
                    _error(field=field_name, code="invalid_date", message="Date is not ISO formatted.")
                )

            if field_name in currency_fields and not isinstance(value, (int, float)):
                record["valid"] = False
                field_errors.append(
                    _error(
                        field=field_name,
                        code="invalid_numeric",
                        message="Currency field must be numeric after normalization.",
                    )
                )

        if field_errors:
            logger.warning(
                "validation_failure",
                extra={"field": field_name, "errors": field_errors},
            )
            errors.extend(field_errors)

    return document, errors


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _preferred_record(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    field_name: str,
    date_fields: set[str],
    currency_fields: set[str],
) -> dict[str, Any]:
    existing_has_value = existing.get("value") not in (None, "")
    candidate_has_value = candidate.get("value") not in (None, "")
    if candidate_has_value and not existing_has_value:
        return candidate
    if existing_has_value and not candidate_has_value:
        return existing

    if field_name in date_fields:
        existing_is_valid = existing_has_value and _is_iso_date(str(existing.get("value")))
        candidate_is_valid = candidate_has_value and _is_iso_date(str(candidate.get("value")))
        if candidate_is_valid and not existing_is_valid:
            return candidate
        if existing_is_valid and not candidate_is_valid:
            return existing

    if field_name in currency_fields:
        existing_is_valid = isinstance(existing.get("value"), (int, float))
        candidate_is_valid = isinstance(candidate.get("value"), (int, float))
        if candidate_is_valid and not existing_is_valid:
            return candidate
        if existing_is_valid and not candidate_is_valid:
            return existing

    return existing if existing["confidence"] >= candidate["confidence"] else candidate


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
