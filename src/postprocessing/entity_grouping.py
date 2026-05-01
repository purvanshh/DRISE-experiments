"""BIO token grouping and entity pairing."""

from __future__ import annotations

import re
from statistics import fmean
from typing import Any


def group_entities(
    predictions: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    spans = _group_bio_spans(predictions)
    grouped_entities: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pending_key: dict[str, Any] | None = None

    for span in spans:
        entity_type = span["entity_type"]
        if entity_type == "KEY":
            if pending_key is not None:
                field_name = canonicalize_field_name(pending_key["text"], field_aliases)
                grouped_entities.append(
                    {
                        "key": pending_key["text"],
                        "field": field_name,
                        "value": None,
                        "confidence": pending_key["confidence"],
                    }
                )
                errors.append(
                    _error(
                        field=field_name,
                        code="missing_value",
                        message=f"Key '{pending_key['text']}' has no paired value.",
                    )
                )
            pending_key = span
            continue

        if entity_type == "VALUE":
            if pending_key is None:
                errors.append(
                    _error(
                        field="_document",
                        code="orphan_value",
                        message=f"Unpaired value '{span['text']}' was ignored.",
                    )
                )
                continue

            field_name = canonicalize_field_name(pending_key["text"], field_aliases)
            grouped_entities.append(
                {
                    "key": pending_key["text"],
                    "field": field_name,
                    "value": span["text"],
                    "confidence": round(fmean([pending_key["confidence"], span["confidence"]]), 6),
                    "key_confidence": pending_key["confidence"],
                    "value_confidence": span["confidence"],
                }
            )
            pending_key = None

    if pending_key is not None:
        field_name = canonicalize_field_name(pending_key["text"], field_aliases)
        grouped_entities.append(
            {
                "key": pending_key["text"],
                "field": field_name,
                "value": None,
                "confidence": pending_key["confidence"],
            }
        )
        errors.append(
            _error(
                field=field_name,
                code="missing_value",
                message=f"Key '{pending_key['text']}' has no paired value.",
            )
        )

    return grouped_entities, errors


def canonicalize_field_name(key_text: str, field_aliases: dict[str, str]) -> str:
    normalized = re.sub(r"[:\s]+", " ", key_text.strip()).lower().strip()
    if normalized in field_aliases:
        return field_aliases[normalized]
    for alias in sorted(field_aliases, key=len, reverse=True):
        if alias in normalized:
            return field_aliases[alias]
    return normalized.replace(" ", "_")


def _group_bio_spans(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for prediction in predictions:
        text = str(prediction.get("text", "")).strip()
        if not text:
            continue

        label = str(prediction.get("label", "O")).upper().strip()
        confidence = float(prediction.get("confidence", 0.0))
        prefix, entity_type = _parse_label(label)

        if entity_type == "OTHER":
            if current is not None:
                spans.append(_finalize_span(current))
                current = None
            continue

        if (
            current is None
            or prefix == "B"
            or current["entity_type"] != entity_type
        ):
            if current is not None:
                spans.append(_finalize_span(current))
            current = {
                "entity_type": entity_type,
                "tokens": [text],
                "confidences": [confidence],
            }
            continue

        current["tokens"].append(text)
        current["confidences"].append(confidence)

    if current is not None:
        spans.append(_finalize_span(current))

    return spans


def _parse_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        return "O", "OTHER"
    prefix, entity_type = label.split("-", maxsplit=1)
    if prefix not in {"B", "I"}:
        return "O", "OTHER"
    if entity_type not in {"KEY", "VALUE"}:
        return "O", "OTHER"
    return prefix, entity_type


def _finalize_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_type": span["entity_type"],
        "text": " ".join(span["tokens"]).strip(),
        "confidence": round(fmean(span["confidences"]), 6),
    }


def _error(field: str, code: str, message: str) -> dict[str, str]:
    return {"field": field, "code": code, "message": message}
