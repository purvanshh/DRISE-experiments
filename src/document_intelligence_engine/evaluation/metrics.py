"""Evaluation metrics.

Includes entity-level seqeval metrics alongside existing exact-match
and field-level accuracy computations.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from datetime import datetime
from decimal import Decimal
import json
import re
from collections import Counter
from typing import Any

from dateutil import parser as date_parser
from jsonschema import Draft202012Validator, FormatChecker
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_exact_match(prediction: dict[str, object], ground_truth: dict[str, object]) -> float:
    return compute_document_exact_match(
        dict(prediction),
        dict(ground_truth),
        tuple(sorted({*prediction.keys(), *ground_truth.keys()})),
    )


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
    schema: dict[str, Any],
) -> float:
    """Return 1.0 when the prediction passes JSON Schema validation."""

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return 1.0 if validator.is_valid(prediction) else 0.0


def compute_hallucination_rate(prediction: dict[str, Any], source_text: str) -> float:
    """Measure the fraction of extracted text-bearing values missing from the source text."""

    normalized_source = _normalize_hallucination_source(source_text)
    normalized_lines = [line for line in (_normalize_hallucination_source(line) for line in source_text.splitlines()) if line]
    checked = 0
    hallucinated = 0

    for field_name, value in prediction.items():
        if str(field_name).startswith("_") or value is None:
            continue
        for candidate_group in _text_hallucination_candidate_groups(str(field_name), value):
            candidate_group = [candidate for candidate in candidate_group if candidate]
            if not candidate_group:
                continue
            checked += 1
            if not any(_text_appears_in_source(candidate, normalized_source, normalized_lines) for candidate in candidate_group):
                hallucinated += 1

    if checked == 0:
        return 0.0
    return hallucinated / checked


def _text_hallucination_candidate_groups(field_name: str, value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = _hallucination_candidates_for_scalar(value, field_name)
        return [candidates] if candidates else []
    if isinstance(value, list):
        candidates: list[list[str]] = []
        for item in value:
            candidates.extend(_text_hallucination_candidate_groups(field_name, item))
        return candidates
    if isinstance(value, dict):
        candidates: list[list[str]] = []
        for nested_key, nested_value in value.items():
            candidates.extend(_text_hallucination_candidate_groups(str(nested_key or field_name), nested_value))
        return candidates
    candidates = _hallucination_candidates_for_scalar(value, field_name)
    return [candidates] if candidates else []


def _text_appears_in_source(candidate: str, normalized_source: str, normalized_lines: list[str]) -> bool:
    if not candidate:
        return True
    if candidate in normalized_source:
        return True
    if len(candidate) < 5:
        return False
    best_ratio = max((SequenceMatcher(None, candidate, line).ratio() for line in normalized_lines), default=0.0)
    return best_ratio >= 0.85


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
        normalized[field] = normalize_field(payload.get(field), field)
    return normalized


def normalize_field(value: Any, field_name: str) -> Any:
    if field_name == "line_items":
        return sorted(
            (_normalize_line_item(item) for item in value or []),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if field_name in {"total_amount", "tax_amount", "subtotal", "unit_price", "price", "quantity"}:
        return _normalize_amount_string(value)
    if field_name == "date":
        return _normalize_date(value)
    if value is None:
        return None
    return _normalize_text_value(value)


def _normalize_line_item(item: Any) -> dict[str, Any] | Any:
    if not isinstance(item, dict):
        return _normalize_text_value(item) if item is not None else None
    normalized: dict[str, Any] = {}
    for key, value in sorted(item.items()):
        normalized[key] = normalize_field(value, key)
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
    try:
        return date_parser.parse(raw, fuzzy=True).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        pass
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if iso_match:
        return iso_match.group(1)
    return None


def _normalize_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = _normalize_string(str(value))
    text = re.sub(r"[^\w\s/-]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _normalize_amount_string(value: Any) -> str | None:
    amount = _normalize_amount(value)
    if amount is None:
        return None
    return f"{Decimal(str(amount)).quantize(Decimal('0.01')):.2f}"


def _normalize_hallucination_source(value: str) -> str:
    normalized = _normalize_string(value)
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"[$€£¥]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _normalize_for_hallucination(value: Any, field_name: str) -> str:
    if value is None or value == "":
        return ""
    if field_name in {"total_amount", "tax_amount", "subtotal", "unit_price", "price", "quantity"}:
        amount = _normalize_amount(value)
        if amount is None:
            return ""
        text = f"{Decimal(str(amount)).quantize(Decimal('0.01')):.2f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    if field_name == "date":
        normalized_date = _normalize_date(value)
        if not normalized_date:
            return ""
        return normalized_date
    normalized_text = _normalize_text_value(value)
    return _normalize_hallucination_source(normalized_text or "")


def _hallucination_candidates_for_scalar(value: Any, field_name: str) -> list[str]:
    if field_name == "date":
        normalized_date = _normalize_date(value)
        if not normalized_date:
            return []
        year, month, day = normalized_date.split("-")
        return [
            _normalize_hallucination_source(normalized_date),
            _normalize_hallucination_source(f"{month}/{day}/{year}"),
            _normalize_hallucination_source(f"{day}/{month}/{year}"),
            _normalize_hallucination_source(f"{month}-{day}-{year}"),
            _normalize_hallucination_source(f"{day}-{month}-{year}"),
        ]
    if field_name in {"total_amount", "tax_amount", "subtotal", "unit_price", "price", "quantity"}:
        amount = _normalize_amount(value)
        if amount is None:
            return []
        canonical = f"{Decimal(str(amount)).quantize(Decimal('0.01')):.2f}"
        trimmed = canonical.rstrip("0").rstrip(".") if "." in canonical else canonical
        return list(dict.fromkeys([canonical, trimmed]))

    normalized = _normalize_for_hallucination(value, field_name)
    return [normalized] if normalized else []
