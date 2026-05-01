"""Field normalization and OCR artifact cleanup."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%b %d %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def normalize_entities(
    entities: list[dict[str, Any]],
    settings: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    normalized_entities: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for entity in entities:
        normalized_entity = dict(entity)
        field_name = normalized_entity["field"]
        original_value = normalized_entity.get("value")

        if original_value is None:
            normalized_entities.append(normalized_entity)
            continue

        if isinstance(original_value, list):
            normalized_value = _normalize_list_value(original_value)
            cleaned_value = str(original_value)
            corrected_value = cleaned_value
        else:
            cleaned_value = cleanup_text(str(original_value))
            corrected_value = cleaned_value

            if field_name in settings.postprocessing.normalization.date_fields:
                corrected_value = fix_ocr_artifacts(
                    cleaned_value,
                    settings.postprocessing.normalization.artifact_map,
                )
                normalized_value = normalize_date(corrected_value)
            elif field_name in settings.postprocessing.normalization.currency_fields:
                corrected_value = fix_ocr_artifacts(
                    cleaned_value,
                    settings.postprocessing.normalization.artifact_map,
                )
                normalized_value = normalize_currency(corrected_value)
            else:
                normalized_value = cleaned_value

        normalized_entity["value"] = normalized_value
        normalized_entities.append(normalized_entity)

        if corrected_value != cleaned_value:
            logger.info(
                "value_corrected",
                extra={"field": field_name, "original": cleaned_value, "corrected": corrected_value},
            )
        if normalized_value != original_value:
            logger.info(
                "value_normalized",
                extra={"field": field_name, "original": original_value, "normalized": normalized_value},
            )
        if normalized_value is None:
            errors.append(
                _error(
                    field=field_name,
                    code="normalization_failed",
                    message=f"Unable to normalize value '{original_value}'.",
                )
            )

    return normalized_entities, errors


def cleanup_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fix_ocr_artifacts(value: str, artifact_map: dict[str, str]) -> str:
    if not value:
        return value
    if not re.fullmatch(r"[0-9A-Za-z$€£¥,.\-:/() ]+", value):
        return value

    characters = list(value)
    for index, character in enumerate(characters):
        if character not in artifact_map:
            continue
        if _numeric_context(value, index):
            characters[index] = artifact_map[character]
    return "".join(characters)


def normalize_date(value: str) -> str | None:
    candidate = cleanup_text(value).replace(",", "")
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_currency(value: str) -> float | None:
    candidate = cleanup_text(value)
    negative = candidate.startswith("(") and candidate.endswith(")")
    candidate = candidate.strip("()")
    candidate = re.sub(r"[^0-9.\-]", "", candidate)
    if candidate.count(".") > 1:
        head, *tail = candidate.split(".")
        candidate = head + "." + "".join(tail)
    if not candidate:
        return None
    try:
        amount = float(Decimal(candidate))
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _normalize_list_value(items: list[Any]) -> list[Any]:
    normalized_items: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        normalized_item: dict[str, Any] = {}
        for key, value in item.items():
            if key in {"quantity", "unit_price", "price"}:
                normalized_item[key] = normalize_currency(str(value)) if value not in (None, "") else None
            else:
                normalized_item[key] = cleanup_text(str(value)) if value not in (None, "") else value
        normalized_items.append(normalized_item)
    return normalized_items


def _numeric_context(value: str, index: int) -> bool:
    current = value[index]
    if current.isdigit():
        return False
    neighbors = []
    if index > 0:
        neighbors.append(value[index - 1])
    if index < len(value) - 1:
        neighbors.append(value[index + 1])
    return any(character.isdigit() for character in neighbors)


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
