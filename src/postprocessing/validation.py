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
                preferred = existing if existing["confidence"] >= confidence else record
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

    regex_rules = settings.postprocessing.validation.regex_rules
    date_fields = set(settings.postprocessing.normalization.date_fields)
    currency_fields = set(settings.postprocessing.normalization.currency_fields)

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


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
