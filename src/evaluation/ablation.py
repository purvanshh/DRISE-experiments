"""Ablation study framework for document intelligence pipelines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from document_intelligence_engine.core.config import get_settings
from evaluation.benchmark import _invoke_model, _invoke_ocr, _invoke_postprocessor, _load_dataset
from evaluation.metrics import compute_structured_output_metrics


def run_ablation_study(dataset_path: str, model: Any, pipeline: Any) -> dict[str, Any]:
    settings = get_settings()
    dataset = _load_dataset(dataset_path)
    experiments = [
        ("remove_layout_embeddings", _without_layout),
        ("remove_postprocessing_layer", _without_postprocessing),
        ("reduce_ocr_quality", _reduce_ocr_quality),
    ]

    results: list[dict[str, Any]] = []
    for experiment_name, experiment_fn in experiments:
        predictions: list[dict[str, Any]] = []
        ground_truths: list[dict[str, Any]] = []
        for sample in dataset:
            ground_truths.append(sample.get("ground_truth", {}))
            predictions.append(experiment_fn(sample["document_path"], model, pipeline))
        metrics = compute_structured_output_metrics(predictions, ground_truths)
        results.append({"experiment": experiment_name, "metrics": metrics})

    threshold_results: list[dict[str, Any]] = []
    for threshold in settings.evaluation.confidence_thresholds:
        predictions = []
        ground_truths = []
        for sample in dataset:
            ocr_tokens = _invoke_ocr(sample["document_path"])
            raw_predictions = _invoke_model(model, "full_system", ocr_tokens)
            output = _invoke_postprocessor(pipeline, raw_predictions)
            output = _apply_confidence_threshold(output, threshold)
            predictions.append(output)
            ground_truths.append(sample.get("ground_truth", {}))
        threshold_results.append(
            {
                "experiment": f"confidence_threshold_{threshold:.2f}",
                "metrics": compute_structured_output_metrics(predictions, ground_truths),
            }
        )

    comparison_table = [
        {
            "experiment": item["experiment"],
            "exact_match_accuracy": item["metrics"]["exact_match_accuracy"],
            "precision": item["metrics"]["precision"],
            "recall": item["metrics"]["recall"],
            "f1": item["metrics"]["f1"],
            "partial_match": item["metrics"]["partial_match"],
        }
        for item in results + threshold_results
    ]

    return {
        "dataset_path": dataset_path,
        "results": results + threshold_results,
        "comparison_table": comparison_table,
    }


def _without_layout(document_path: str, model: Any, pipeline: Any) -> dict[str, Any]:
    ocr_tokens = _invoke_ocr(document_path)
    raw_predictions = _invoke_model(model, "text_only_model", ocr_tokens)
    return _invoke_postprocessor(pipeline, raw_predictions)


def _without_postprocessing(document_path: str, model: Any, pipeline: Any) -> dict[str, Any]:
    _ = pipeline
    settings = get_settings()
    ocr_tokens = _invoke_ocr(document_path)
    raw_predictions = _invoke_model(model, "layoutlmv3_no_postprocessing", ocr_tokens)
    from evaluation.benchmark import _raw_predictions_to_structured

    return _raw_predictions_to_structured(raw_predictions, settings.postprocessing.field_aliases)


def _reduce_ocr_quality(document_path: str, model: Any, pipeline: Any) -> dict[str, Any]:
    ocr_tokens = _invoke_ocr(document_path)
    noisy_tokens = [_degrade_token(token, index) for index, token in enumerate(ocr_tokens)]
    raw_predictions = _invoke_model(model, "full_system", noisy_tokens)
    return _invoke_postprocessor(pipeline, raw_predictions)


def _degrade_token(token: dict[str, Any], index: int) -> dict[str, Any]:
    degraded = deepcopy(token)
    confidence = float(degraded.get("confidence", 0.0))
    degraded["confidence"] = max(0.0, round(confidence - 0.25, 6))
    if index % 4 == 0:
        degraded["text"] = str(degraded.get("text", ""))[:-1]
    return degraded


def _apply_confidence_threshold(document: dict[str, Any], threshold: float) -> dict[str, Any]:
    filtered = {}
    errors = list(document.get("_errors", []))
    for field_name, payload in document.items():
        if field_name.startswith("_"):
            continue
        if isinstance(payload, dict) and float(payload.get("confidence", 0.0)) >= threshold:
            filtered[field_name] = payload
        elif isinstance(payload, dict):
            errors.append(
                {
                    "field": field_name,
                    "code": "ablation_dropped_low_confidence",
                    "message": f"Field dropped at threshold {threshold:.2f}.",
                }
            )
    filtered["_errors"] = errors
    filtered["_constraint_flags"] = list(document.get("_constraint_flags", []))
    return filtered
