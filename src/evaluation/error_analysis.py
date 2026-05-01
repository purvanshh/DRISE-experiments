"""Error categorization and sample harvesting."""

from __future__ import annotations

from typing import Any

from document_intelligence_engine.core.config import get_settings


def analyze_errors(benchmark_results: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    categories: dict[str, dict[str, Any]] = {
        "ocr_errors": {"count": 0, "examples": []},
        "layout_errors": {"count": 0, "examples": []},
        "model_misclassification": {"count": 0, "examples": []},
        "postprocessing_failures": {"count": 0, "examples": []},
    }

    for sample in benchmark_results.get("sample_results", []):
        full_metrics = sample["metrics"]["full_system"]["structured_output"]
        layout_metrics = sample["metrics"]["layoutlmv3_no_postprocessing"]["structured_output"]
        text_metrics = sample["metrics"]["text_only_model"]["structured_output"]
        ocr_metrics = sample["metrics"]["ocr_only"]["structured_output"]

        if sample.get("ocr_token_count", 0) == 0 or ocr_metrics.get("partial_match", 0.0) == 0.0:
            _append_category(categories, "ocr_errors", sample, settings.evaluation.error_example_limit)

        if text_metrics.get("f1", 0.0) > layout_metrics.get("f1", 0.0):
            _append_category(categories, "layout_errors", sample, settings.evaluation.error_example_limit)

        if layout_metrics.get("partial_match", 0.0) < text_metrics.get("partial_match", 0.0):
            _append_category(categories, "model_misclassification", sample, settings.evaluation.error_example_limit)

        if layout_metrics.get("f1", 0.0) > full_metrics.get("f1", 0.0):
            _append_category(categories, "postprocessing_failures", sample, settings.evaluation.error_example_limit)

    return categories


def _append_category(
    categories: dict[str, dict[str, Any]],
    category: str,
    sample: dict[str, Any],
    limit: int,
) -> None:
    categories[category]["count"] += 1
    if len(categories[category]["examples"]) < limit:
        categories[category]["examples"].append(
            {
                "sample_id": sample.get("sample_id"),
                "document_path": sample.get("document_path"),
                "ground_truth": sample.get("ground_truth"),
                "baseline_outputs": sample.get("baseline_outputs"),
            }
        )
