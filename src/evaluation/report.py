"""Report generation for benchmark and ablation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from document_intelligence_engine.core.config import get_settings
from evaluation.error_analysis import analyze_errors


def generate_report(
    benchmark_results: dict[str, Any],
    ablation_results: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    report_dir = Path(output_dir or settings.evaluation.output_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    ablations = ablation_results or {"results": [], "comparison_table": []}
    error_breakdown = analyze_errors(benchmark_results)
    plots = _generate_plots(benchmark_results, report_dir)

    report_payload = {
        "summary": {
            "dataset_path": benchmark_results["dataset_path"],
            "sample_count": benchmark_results["sample_count"],
            "primary_metrics": benchmark_results["metrics"],
        },
        "baselines": benchmark_results["baselines"],
        "ablations": ablations,
        "errors": error_breakdown,
        "tracking": benchmark_results.get("tracking", {}),
        "artifacts": plots,
    }

    json_path = report_dir / "benchmark_report.json"
    markdown_path = report_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_report(report_payload), encoding="utf-8")

    return {
        "report": report_payload,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "artifacts": plots,
    }


def _render_markdown_report(report_payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        f"- Dataset: `{report_payload['summary']['dataset_path']}`",
        f"- Samples: `{report_payload['summary']['sample_count']}`",
        "",
        "## Baselines",
    ]
    for baseline_name, metrics in report_payload["baselines"].items():
        structured = metrics.get("structured_output", {})
        lines.append(
            f"- {baseline_name}: exact_match={structured.get('exact_match_accuracy', 0.0):.4f}, "
            f"f1={structured.get('f1', 0.0):.4f}, partial_match={structured.get('partial_match', 0.0):.4f}"
        )
    lines.extend(["", "## Ablations"])
    for result in report_payload["ablations"].get("comparison_table", []):
        lines.append(
            f"- {result['experiment']}: exact_match={result['exact_match_accuracy']:.4f}, "
            f"f1={result['f1']:.4f}, partial_match={result['partial_match']:.4f}"
        )
    lines.extend(["", "## Error Breakdown"])
    for category, details in report_payload["errors"].items():
        lines.append(f"- {category}: count={details['count']}")
    return "\n".join(lines) + "\n"


def _generate_plots(benchmark_results: dict[str, Any], report_dir: Path) -> dict[str, str]:
    baseline_names = []
    exact_match_scores = []
    for baseline_name, metrics in benchmark_results["baselines"].items():
        baseline_names.append(baseline_name)
        exact_match_scores.append(metrics["structured_output"].get("exact_match_accuracy", 0.0))

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(baseline_names, exact_match_scores)
    axis.set_ylabel("Exact Match Accuracy")
    axis.set_title("Baseline Comparison")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    comparison_plot_path = report_dir / "baseline_comparison.png"
    figure.savefig(comparison_plot_path)
    plt.close(figure)
    return {"baseline_comparison_plot": str(comparison_plot_path)}
