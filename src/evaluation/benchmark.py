"""Benchmark runner for document intelligence pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.logging import get_logger
from evaluation.metrics import (
    compute_entity_metrics,
    compute_structured_output_metrics,
    compute_token_metrics,
)
from ingestion.pipeline import process_document
from postprocessing.entity_grouping import group_entities


logger = get_logger(__name__)


def run_benchmark(dataset_path: str, model: Any, pipeline: Any) -> dict[str, Any]:
    settings = get_settings()
    dataset = _load_dataset(dataset_path)
    baseline_predictions: dict[str, list[dict[str, Any]]] = {
        "ocr_only": [],
        "text_only_model": [],
        "layoutlmv3_no_postprocessing": [],
        "full_system": [],
    }
    ground_truth_outputs: list[dict[str, Any]] = []
    token_truth: list[list[str]] = []
    token_predictions: dict[str, list[list[str]]] = {name: [] for name in baseline_predictions}
    entity_truth: list[list[dict[str, Any]]] = []
    entity_predictions: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in baseline_predictions}
    sample_results: list[dict[str, Any]] = []

    for sample in dataset:
        document_path = sample["document_path"]
        ocr_tokens = _invoke_ocr(document_path)
        raw_prediction_full = _invoke_model(model, "full_system", ocr_tokens)
        raw_prediction_text_only = _invoke_model(model, "text_only_model", ocr_tokens)
        raw_prediction_layout = _invoke_model(model, "layoutlmv3_no_postprocessing", ocr_tokens)
        ocr_only_output = _run_ocr_only_baseline(ocr_tokens, settings.postprocessing.field_aliases)
        text_only_output = _invoke_postprocessor(pipeline, raw_prediction_text_only)
        layout_no_post_output = _raw_predictions_to_structured(
            raw_prediction_layout,
            settings.postprocessing.field_aliases,
        )
        full_system_output = _invoke_postprocessor(pipeline, raw_prediction_full)

        baseline_outputs = {
            "ocr_only": ocr_only_output,
            "text_only_model": text_only_output,
            "layoutlmv3_no_postprocessing": layout_no_post_output,
            "full_system": full_system_output,
        }

        sample_ground_truth = sample.get("ground_truth", {})
        sample_entities = sample.get("entities", _structured_to_entities(sample_ground_truth))
        sample_token_labels = sample.get("token_labels", [])

        ground_truth_outputs.append(sample_ground_truth)
        entity_truth.append(sample_entities)
        token_truth.append(sample_token_labels)

        per_sample_metrics: dict[str, Any] = {}
        for baseline_name, output in baseline_outputs.items():
            baseline_predictions[baseline_name].append(output)
            raw_predictions = _baseline_raw_predictions(
                baseline_name,
                raw_prediction_full,
                raw_prediction_text_only,
                raw_prediction_layout,
                ocr_tokens,
            )
            token_labels = [str(item.get("label", "O")) for item in raw_predictions]
            token_predictions[baseline_name].append(token_labels)
            predicted_entities = _raw_predictions_to_entities(raw_predictions, settings.postprocessing.field_aliases)
            entity_predictions[baseline_name].append(predicted_entities)
            per_sample_metrics[baseline_name] = {
                "token_level": compute_token_metrics(sample_token_labels, token_labels) if sample_token_labels else {},
                "entity_level": compute_entity_metrics(sample_entities, predicted_entities),
                "structured_output": compute_structured_output_metrics(output, sample_ground_truth),
            }

        sample_results.append(
            {
                "sample_id": sample.get("id", Path(document_path).stem),
                "document_path": document_path,
                "ground_truth": sample_ground_truth,
                "ocr_token_count": len(ocr_tokens),
                "baseline_outputs": baseline_outputs,
                "raw_predictions": {
                    "text_only_model": raw_prediction_text_only,
                    "layoutlmv3_no_postprocessing": raw_prediction_layout,
                    "full_system": raw_prediction_full,
                },
                "metrics": per_sample_metrics,
            }
        )

    aggregated_metrics = {
        baseline_name: _aggregate_metrics(
            token_truth,
            token_predictions[baseline_name],
            entity_truth,
            entity_predictions[baseline_name],
            ground_truth_outputs,
            baseline_predictions[baseline_name],
        )
        for baseline_name in baseline_predictions
    }

    benchmark_result = {
        "dataset_path": dataset_path,
        "sample_count": len(dataset),
        "metrics": aggregated_metrics["full_system"],
        "baselines": aggregated_metrics,
        "sample_results": sample_results,
        "tracking": _track_results(settings, model, dataset_path, aggregated_metrics),
    }
    benchmark_result["tracking"]["logged"] = benchmark_result["tracking"].get("logged", False)
    return benchmark_result


def _aggregate_metrics(
    token_truth: list[list[str]],
    token_predictions: list[list[str]],
    entity_truth: list[list[dict[str, Any]]],
    entity_predictions: list[list[dict[str, Any]]],
    ground_truth_outputs: list[dict[str, Any]],
    predicted_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "token_level": compute_token_metrics(token_truth, token_predictions) if token_truth and token_predictions else {},
        "entity_level": compute_entity_metrics(entity_truth, entity_predictions),
        "structured_output": compute_structured_output_metrics(predicted_outputs, ground_truth_outputs),
    }


def _track_results(
    settings: Any,
    model: Any,
    dataset_path: str,
    aggregated_metrics: dict[str, Any],
) -> dict[str, Any]:
    tracking_settings = settings.evaluation.tracking
    if not tracking_settings.enabled:
        return {"logged": False}

    if tracking_settings.uri:
        mlflow.set_tracking_uri(tracking_settings.uri)
    mlflow.set_experiment(tracking_settings.experiment_name)
    with mlflow.start_run(run_name="benchmark") as run:
        mlflow.log_params(
            {
                "dataset_path": dataset_path,
                "model_name": getattr(model, "name", model.__class__.__name__),
                "ocr_backend": settings.ocr.backend,
                "postprocessing_threshold": settings.postprocessing.confidence.min_field_confidence,
            }
        )
        for baseline_name, metrics in aggregated_metrics.items():
            structured_metrics = metrics.get("structured_output", {})
            mlflow.log_metric(f"{baseline_name}_exact_match_accuracy", structured_metrics.get("exact_match_accuracy", 0.0))
            mlflow.log_metric(f"{baseline_name}_f1", structured_metrics.get("f1", 0.0))
        return {"logged": True, "run_id": run.info.run_id}


def _load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file_pointer:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file_pointer if line.strip()]
        payload = json.load(file_pointer)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError("Unsupported dataset format.")


def _invoke_ocr(document_path: str) -> list[dict[str, Any]]:
    logger.info("benchmark_sample_started", extra={"document_path": document_path})
    return process_document(document_path)


def _invoke_model(model: Any, mode: str, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if mode == "text_only_model":
        method_names = ("predict_text_only", "predict_without_layout", "predict")
    elif mode == "layoutlmv3_no_postprocessing":
        method_names = ("predict_without_postprocessing", "predict_layout", "predict")
    else:
        method_names = ("predict",)

    for method_name in method_names:
        method = getattr(model, method_name, None)
        if callable(method):
            return method(ocr_tokens)
    if callable(model):
        return model(ocr_tokens)
    raise AttributeError(f"Model does not implement a callable method for mode '{mode}'.")


def _invoke_postprocessor(pipeline: Any, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if callable(pipeline):
        return pipeline(predictions)
    if hasattr(pipeline, "postprocess_predictions"):
        return pipeline.postprocess_predictions(predictions)
    if hasattr(pipeline, "process"):
        return pipeline.process(predictions)
    raise AttributeError("Pipeline does not expose a postprocessing entrypoint.")


def _run_ocr_only_baseline(
    ocr_tokens: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    pending_key: str | None = None
    for token in ocr_tokens:
        text = str(token.get("text", "")).strip()
        if not text:
            continue
        lower_text = text.lower().rstrip(":")
        if pending_key is None and lower_text in field_aliases:
            pending_key = field_aliases[lower_text]
            continue
        if pending_key is not None:
            output[pending_key] = {
                "value": text,
                "confidence": round(float(token.get("confidence", 0.0)), 6),
                "valid": True,
            }
            pending_key = None
    output.setdefault("_errors", [])
    output.setdefault("_constraint_flags", [])
    return output


def _raw_predictions_to_structured(
    predictions: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> dict[str, Any]:
    grouped_entities, errors = group_entities(predictions, field_aliases)
    structured: dict[str, Any] = {}
    for entity in grouped_entities:
        structured[entity["field"]] = {
            "value": entity.get("value"),
            "confidence": round(float(entity.get("confidence", 0.0)), 6),
            "valid": entity.get("value") is not None,
        }
    structured["_errors"] = errors
    structured["_constraint_flags"] = []
    return structured


def _raw_predictions_to_entities(
    predictions: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> list[dict[str, Any]]:
    grouped_entities, _ = group_entities(predictions, field_aliases)
    return grouped_entities


def _structured_to_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for field_name, value in payload.items():
        if field_name.startswith("_"):
            continue
        if isinstance(value, dict) and "value" in value:
            entity_value = value["value"]
        else:
            entity_value = value
        entities.append({"field": field_name, "value": entity_value})
    return entities


def _baseline_raw_predictions(
    baseline_name: str,
    raw_prediction_full: list[dict[str, Any]],
    raw_prediction_text_only: list[dict[str, Any]],
    raw_prediction_layout: list[dict[str, Any]],
    ocr_tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if baseline_name == "text_only_model":
        return raw_prediction_text_only
    if baseline_name == "layoutlmv3_no_postprocessing":
        return raw_prediction_layout
    if baseline_name == "full_system":
        return raw_prediction_full
    return [{"text": token.get("text", ""), "label": "O", "confidence": token.get("confidence", 0.0)} for token in ocr_tokens]
