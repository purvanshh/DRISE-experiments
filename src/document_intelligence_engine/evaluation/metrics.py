"""Evaluation metrics.

Includes entity-level seqeval metrics alongside existing exact-match
and field-level accuracy computations.
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from collections import Counter
from typing import Any

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_exact_match(prediction: dict[str, object], ground_truth: dict[str, object]) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def compute_field_level_accuracy(
    predictions: list[dict[str, object]],
    ground_truths: list[dict[str, object]],
    fields: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not predictions or not ground_truths:
        return {field: 0.0 for field in fields}

    sample_count = min(len(predictions), len(ground_truths))
    for field in fields:
        matches = 0
        for index in range(sample_count):
            if predictions[index].get(field) == ground_truths[index].get(field):
                matches += 1
        results[field] = matches / sample_count
    return results


def compute_entity_f1(
    pred_labels: list[list[str]],
    true_labels: list[list[str]],
    average: str = "micro",
) -> float:
    """Compute entity-level F1 using seqeval.

    Args:
        pred_labels: List of predicted label sequences (one per sample).
        true_labels: List of ground-truth label sequences (one per sample).
        average: Averaging mode — 'micro', 'macro', or 'weighted'.

    Returns:
        F1 score as a float.
    """
    return f1_score(true_labels, pred_labels, average=average, zero_division=0)


def compute_entity_precision_recall(
    pred_labels: list[list[str]],
    true_labels: list[list[str]],
    average: str = "micro",
) -> dict[str, float]:
    """Compute entity-level precision, recall, and F1.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    return {
        "precision": precision_score(true_labels, pred_labels, average=average, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average=average, zero_division=0),
        "f1": f1_score(true_labels, pred_labels, average=average, zero_division=0),
    }


def compute_entity_report(
    pred_labels: list[list[str]],
    true_labels: list[list[str]],
) -> str:
    """Return a full classification report string from seqeval."""
    return classification_report(true_labels, pred_labels, zero_division=0)


DEFAULT_FIELDS = ("invoice_number", "date", "vendor", "total_amount", "line_items")


def compute_document_exact_match(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    fields: list[str] | tuple[str, ...] = DEFAULT_FIELDS,
) -> float:
    """Compute normalized document-level exact match across key fields."""

    return 1.0 if _normalized_field_payload(prediction, fields) == _normalized_field_payload(ground_truth, fields) else 0.0


def compute_field_f1_scores(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    fields: list[str] | tuple[str, ...] = DEFAULT_FIELDS,
) -> dict[str, float]:
    """Compute token-level F1 for each field independently."""

    return {field: _field_f1(prediction.get(field), ground_truth.get(field)) for field in fields}


def compute_field_level_f1(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    fields: list[str] | tuple[str, ...] = DEFAULT_FIELDS,
) -> float:
    """Compute a micro-style F1 across all configured fields."""

    total_true_positive = 0
    total_predicted = 0
    total_ground_truth = 0

    for field in fields:
        true_positive, predicted_count, ground_truth_count = _token_overlap_counts(
            prediction.get(field),
            ground_truth.get(field),
        )
        total_true_positive += true_positive
        total_predicted += predicted_count
        total_ground_truth += ground_truth_count

    if total_predicted == 0 and total_ground_truth == 0:
        return 1.0
    if total_predicted == 0 or total_ground_truth == 0:
        return 0.0

    precision = total_true_positive / total_predicted
    recall = total_true_positive / total_ground_truth
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_schema_validity(
    prediction: dict[str, Any],
    schema: dict[str, type | tuple[type, ...]],
) -> float:
    """Return 1.0 when the prediction matches required field presence and types."""

    for field_name, expected_type in schema.items():
        if field_name not in prediction:
            return 0.0
        value = prediction[field_name]
        if value is None:
            continue
        if not isinstance(value, expected_type):
            return 0.0
        if field_name == "line_items" and any(not isinstance(item, dict) for item in value):
            return 0.0
    return 1.0


def compute_hallucination_rate(prediction: dict[str, Any], source_text: str) -> float:
    """Measure the fraction of extracted scalar fields missing from the source text."""

    normalized_source = _normalize_string(source_text)
    checked = 0
    hallucinated = 0

    for field_name, value in prediction.items():
        if str(field_name).startswith("_") or value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, list):
            values_to_check = [_normalize_string(_serialize_line_item(item)) for item in value if item]
        else:
            values_to_check = [_normalize_string(_normalize_value(value))]

        for candidate in values_to_check:
            if not candidate:
                continue
            checked += 1
            if candidate not in normalized_source:
                hallucinated += 1

    if checked == 0:
        return 0.0
    return hallucinated / checked


def _field_f1(prediction: Any, ground_truth: Any) -> float:
    true_positive, predicted_count, ground_truth_count = _token_overlap_counts(prediction, ground_truth)
    if predicted_count == 0 and ground_truth_count == 0:
        return 1.0
    if predicted_count == 0 or ground_truth_count == 0:
        return 0.0
    precision = true_positive / predicted_count
    recall = true_positive / ground_truth_count
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _token_overlap_counts(prediction: Any, ground_truth: Any) -> tuple[int, int, int]:
    prediction_tokens = Counter(_value_tokens(prediction))
    ground_truth_tokens = Counter(_value_tokens(ground_truth))
    intersection = prediction_tokens & ground_truth_tokens
    return sum(intersection.values()), sum(prediction_tokens.values()), sum(ground_truth_tokens.values())


def _value_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_value_tokens(item))
        return tokens
    if isinstance(value, dict):
        ordered_parts = []
        for key in sorted(value):
            ordered_parts.append(_normalize_value(value[key]))
        serialized = " ".join(part for part in ordered_parts if part)
        return re.findall(r"[a-z0-9.]+", serialized.lower())

    normalized = _normalize_value(value)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9.]+", normalized.lower())


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    if isinstance(value, list):
        return " ".join(_normalize_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value).strip()


def _normalized_field_payload(payload: dict[str, Any], fields: list[str] | tuple[str, ...]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in fields:
        value = payload.get(field)
        if field == "line_items":
            normalized[field] = sorted(
                (_normalize_line_item(item) for item in value or []),
                key=lambda item: json.dumps(item, sort_keys=True, default=str),
            )
        elif field == "total_amount":
            normalized[field] = _normalize_amount(value)
        elif field == "date":
            normalized[field] = _normalize_date(value)
        elif value is None:
            normalized[field] = None
        else:
            normalized[field] = str(value).strip()
    return normalized


def _normalize_line_item(item: Any) -> dict[str, Any] | Any:
    if not isinstance(item, dict):
        return item
    normalized: dict[str, Any] = {}
    for key, value in sorted(item.items()):
        if key in {"quantity", "unit_price", "price"}:
            normalized[key] = _normalize_amount(value)
        elif key == "date":
            normalized[key] = _normalize_date(value)
        elif value is None:
            normalized[key] = None
        else:
            normalized[key] = str(value).strip()
    return normalized


def _serialize_line_item(item: Any) -> str:
    if isinstance(item, dict):
        return " ".join(_normalize_value(item.get(key)) for key in sorted(item))
    return _normalize_value(item)


def _normalize_string(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_amount(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    cleaned = re.sub(r"[^0-9.\-]+", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _normalize_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    raw = str(value).strip()
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if iso_match:
        return iso_match.group(1)
    return raw
