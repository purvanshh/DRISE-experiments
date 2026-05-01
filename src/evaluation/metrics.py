"""Evaluation metrics for token, entity, and structured outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from seqeval.metrics import f1_score, precision_score, recall_score


def compute_token_metrics(
    true_labels: list[list[str]] | list[str],
    predicted_labels: list[list[str]] | list[str],
) -> dict[str, float]:
    true_sequences = _ensure_sequences(true_labels)
    predicted_sequences = _ensure_sequences(predicted_labels)
    if not true_sequences or not predicted_sequences:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(float(precision_score(true_sequences, predicted_sequences)), 6),
        "recall": round(float(recall_score(true_sequences, predicted_sequences)), 6),
        "f1": round(float(f1_score(true_sequences, predicted_sequences)), 6),
    }


def compute_entity_metrics(
    true_entities: list[list[dict[str, Any]]] | list[dict[str, Any]],
    predicted_entities: list[list[dict[str, Any]]] | list[dict[str, Any]],
) -> dict[str, float]:
    true_batches = _ensure_batches(true_entities)
    predicted_batches = _ensure_batches(predicted_entities)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_batch, predicted_batch in zip(true_batches, predicted_batches, strict=False):
        true_set = {_entity_signature(entity) for entity in true_batch}
        predicted_set = {_entity_signature(entity) for entity in predicted_batch}
        true_positive += len(true_set & predicted_set)
        false_positive += len(predicted_set - true_set)
        false_negative += len(true_set - predicted_set)

    precision, recall, f1_value = _precision_recall_f1(true_positive, false_positive, false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_value,
    }


def compute_structured_output_metrics(
    predictions: list[dict[str, Any]] | dict[str, Any],
    ground_truths: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    prediction_batch = predictions if isinstance(predictions, list) else [predictions]
    ground_truth_batch = ground_truths if isinstance(ground_truths, list) else [ground_truths]

    exact_match_count = 0
    total_documents = min(len(prediction_batch), len(ground_truth_batch))
    field_stats: dict[str, dict[str, float]] = {}
    partial_scores: list[float] = []

    for predicted, ground_truth in zip(prediction_batch, ground_truth_batch, strict=False):
        cleaned_predicted = _structured_pairs(predicted)
        cleaned_ground_truth = _structured_pairs(ground_truth)
        if cleaned_predicted == cleaned_ground_truth:
            exact_match_count += 1

        for field_name in set(cleaned_predicted) | set(cleaned_ground_truth):
            stats = field_stats.setdefault(
                field_name,
                {"matches": 0.0, "predicted": 0.0, "ground_truth": 0.0, "partial_score": 0.0, "count": 0.0},
            )
            pred_value = cleaned_predicted.get(field_name)
            true_value = cleaned_ground_truth.get(field_name)
            if pred_value is not None:
                stats["predicted"] += 1
            if true_value is not None:
                stats["ground_truth"] += 1
            if pred_value is not None and true_value is not None:
                similarity = partial_match_score(pred_value, true_value)
                partial_scores.append(similarity)
                stats["partial_score"] += similarity
                stats["count"] += 1
                if pred_value == true_value:
                    stats["matches"] += 1

    field_level_accuracy: dict[str, dict[str, float]] = {}
    global_true_positive = 0
    global_false_positive = 0
    global_false_negative = 0

    for field_name, stats in field_stats.items():
        true_positive = int(stats["matches"])
        false_positive = int(max(stats["predicted"] - stats["matches"], 0))
        false_negative = int(max(stats["ground_truth"] - stats["matches"], 0))
        precision, recall, f1_value = _precision_recall_f1(true_positive, false_positive, false_negative)
        accuracy = round(stats["matches"] / stats["ground_truth"], 6) if stats["ground_truth"] else 0.0
        avg_partial = round(stats["partial_score"] / stats["count"], 6) if stats["count"] else 0.0
        field_level_accuracy[field_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "partial_match": avg_partial,
        }
        global_true_positive += true_positive
        global_false_positive += false_positive
        global_false_negative += false_negative

    precision, recall, f1_value = _precision_recall_f1(
        global_true_positive,
        global_false_positive,
        global_false_negative,
    )
    exact_match_accuracy = round(exact_match_count / total_documents, 6) if total_documents else 0.0
    avg_partial_match = round(sum(partial_scores) / len(partial_scores), 6) if partial_scores else 0.0
    return {
        "exact_match_accuracy": exact_match_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_value,
        "partial_match": avg_partial_match,
        "field_level_accuracy": field_level_accuracy,
    }


def partial_match_score(predicted_value: Any, true_value: Any) -> float:
    predicted_text = _stringify_value(predicted_value)
    true_text = _stringify_value(true_value)
    if not predicted_text and not true_text:
        return 1.0
    if not predicted_text or not true_text:
        return 0.0
    return round(SequenceMatcher(None, predicted_text, true_text).ratio(), 6)


def _ensure_sequences(labels: list[list[str]] | list[str]) -> list[list[str]]:
    if not labels:
        return []
    if isinstance(labels[0], str):
        return [list(labels)]  # type: ignore[list-item]
    return labels  # type: ignore[return-value]


def _ensure_batches(
    entities: list[list[dict[str, Any]]] | list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not entities:
        return []
    if isinstance(entities[0], dict):
        return [entities]  # type: ignore[list-item]
    return entities  # type: ignore[return-value]


def _entity_signature(entity: dict[str, Any]) -> tuple[str, str]:
    field_name = str(entity.get("field") or entity.get("key") or "").strip().lower()
    value = _stringify_value(entity.get("value"))
    return field_name, value


def _structured_pairs(payload: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for field_name, value in payload.items():
        if str(field_name).startswith("_"):
            continue
        if isinstance(value, dict) and "value" in value:
            normalized = _stringify_value(value.get("value"))
        else:
            normalized = _stringify_value(value)
        if normalized:
            cleaned[str(field_name)] = normalized
    return cleaned


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "|".join(_stringify_value(item) for item in value)
    if isinstance(value, dict):
        return "|".join(f"{key}:{_stringify_value(val)}" for key, val in sorted(value.items()))
    return str(value).strip()


def _precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    precision = round(true_positive / (true_positive + false_positive), 6) if (true_positive + false_positive) else 0.0
    recall = round(true_positive / (true_positive + false_negative), 6) if (true_positive + false_negative) else 0.0
    f1_value = round((2 * precision * recall) / (precision + recall), 6) if (precision + recall) else 0.0
    return precision, recall, f1_value
