"""Evaluation metrics.

Includes entity-level seqeval metrics alongside existing exact-match
and field-level accuracy computations.
"""

from __future__ import annotations

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
