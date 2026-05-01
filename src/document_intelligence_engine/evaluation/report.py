"""Report generation for DRISE experiments."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

from document_intelligence_engine.evaluation.stats import mcnemar_test

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def generate_experiment_report(
    results_by_system: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path = "experiments/results",
) -> dict[str, Any]:
    """Aggregate experiment results and export summary artifacts."""

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = {system_name: _summarize_system(records) for system_name, records in results_by_system.items()}
    pairwise_stats = _pairwise_stats(results_by_system)
    ablation_summary = _ablation_summary(summary)

    summary_csv = target_dir / "summary.csv"
    summary_json = target_dir / "summary.json"
    summary_md = target_dir / "summary.md"
    stats_json = target_dir / "pairwise_stats.json"
    ablation_csv = target_dir / "ablation_summary.csv"
    ablation_json = target_dir / "ablation_summary.json"

    _write_summary_csv(summary_csv, summary)
    _write_ablation_csv(ablation_csv, ablation_summary)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary, pairwise_stats), encoding="utf-8")
    stats_json.write_text(json.dumps(pairwise_stats, indent=2), encoding="utf-8")
    ablation_json.write_text(json.dumps(ablation_summary, indent=2), encoding="utf-8")

    artifacts = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "pairwise_stats_json": str(stats_json),
        "ablation_summary_csv": str(ablation_csv),
        "ablation_summary_json": str(ablation_json),
    }
    chart_path = _write_field_chart(results_by_system, target_dir)
    if chart_path is not None:
        artifacts["per_field_f1_chart"] = str(chart_path)

    report = {
        "summary": summary,
        "pairwise_stats": pairwise_stats,
        "ablation_summary": ablation_summary,
        "artifacts": artifacts,
    }
    (target_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summarize_system(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [record["metrics"] for record in records]
    field_names = sorted({field for metrics in metric_rows for field in metrics.get("field_f1", {})})
    return {
        "field_level_f1": _mean_and_std(metric_rows, "field_level_f1"),
        "exact_match": _mean_and_std(metric_rows, "exact_match"),
        "schema_valid": _mean_and_std(metric_rows, "schema_valid"),
        "hallucination_rate": _mean_and_std(metric_rows, "hallucination_rate"),
        "latency_ms": _mean_and_std(metric_rows, "latency_ms"),
        "cost_usd": _mean_and_std(metric_rows, "cost_usd"),
        "field_f1_breakdown": {
            field: _mean_and_std_from_values([metrics.get("field_f1", {}).get(field, 0.0) for metrics in metric_rows])
            for field in field_names
        },
        "sample_count": len(metric_rows),
    }


def _pairwise_stats(results_by_system: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    system_names = list(results_by_system)
    comparisons: dict[str, Any] = {}
    for index, left_name in enumerate(system_names):
        left_records = results_by_system[left_name]
        left_correct = [bool(record["metrics"].get("exact_match")) for record in left_records]
        for right_name in system_names[index + 1 :]:
            right_records = results_by_system[right_name]
            right_correct = [bool(record["metrics"].get("exact_match")) for record in right_records]
            comparisons[f"{left_name}_vs_{right_name}"] = mcnemar_test(left_correct, right_correct)
    return comparisons


def _ablation_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = summary.get("drise")
    if baseline is None:
        return []

    variants = []
    for system_name, metrics in summary.items():
        if system_name == "drise" or not system_name.startswith("drise_"):
            continue
        variants.append(
            {
                "system": system_name,
                "field_level_f1_mean": metrics["field_level_f1"]["mean"],
                "exact_match_mean": metrics["exact_match"]["mean"],
                "schema_valid_mean": metrics["schema_valid"]["mean"],
                "hallucination_rate_mean": metrics["hallucination_rate"]["mean"],
                "delta_field_level_f1": round(metrics["field_level_f1"]["mean"] - baseline["field_level_f1"]["mean"], 6),
                "delta_exact_match": round(metrics["exact_match"]["mean"] - baseline["exact_match"]["mean"], 6),
                "delta_schema_valid": round(metrics["schema_valid"]["mean"] - baseline["schema_valid"]["mean"], 6),
                "delta_hallucination_rate": round(
                    metrics["hallucination_rate"]["mean"] - baseline["hallucination_rate"]["mean"],
                    6,
                ),
            }
        )
    return variants


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_pointer:
        writer = csv.writer(file_pointer)
        writer.writerow(
            [
                "system",
                "field_level_f1_mean",
                "field_level_f1_std",
                "exact_match_mean",
                "schema_valid_mean",
                "hallucination_rate_mean",
                "latency_ms_mean",
                "cost_usd_mean",
                "sample_count",
            ]
        )
        for system_name, metrics in summary.items():
            writer.writerow(
                [
                    system_name,
                    metrics["field_level_f1"]["mean"],
                    metrics["field_level_f1"]["std"],
                    metrics["exact_match"]["mean"],
                    metrics["schema_valid"]["mean"],
                    metrics["hallucination_rate"]["mean"],
                    metrics["latency_ms"]["mean"],
                    metrics["cost_usd"]["mean"],
                    metrics["sample_count"],
                ]
            )


def _write_ablation_csv(path: Path, ablations: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_pointer:
        writer = csv.writer(file_pointer)
        writer.writerow(
            [
                "system",
                "field_level_f1_mean",
                "exact_match_mean",
                "schema_valid_mean",
                "hallucination_rate_mean",
                "delta_field_level_f1",
                "delta_exact_match",
                "delta_schema_valid",
                "delta_hallucination_rate",
            ]
        )
        for row in ablations:
            writer.writerow(
                [
                    row["system"],
                    row["field_level_f1_mean"],
                    row["exact_match_mean"],
                    row["schema_valid_mean"],
                    row["hallucination_rate_mean"],
                    row["delta_field_level_f1"],
                    row["delta_exact_match"],
                    row["delta_schema_valid"],
                    row["delta_hallucination_rate"],
                ]
            )


def _render_markdown(summary: dict[str, Any], pairwise_stats: dict[str, Any]) -> str:
    lines = [
        "## Experimental Results",
        "",
        "| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for system_name, metrics in summary.items():
        lines.append(
            "| {system} | {f1:.3f} | {exact:.3f} | {schema:.3f} | {hallucination:.3f} | ${cost:.4f} |".format(
                system=system_name,
                f1=metrics["field_level_f1"]["mean"],
                exact=metrics["exact_match"]["mean"],
                schema=metrics["schema_valid"]["mean"],
                hallucination=metrics["hallucination_rate"]["mean"],
                cost=metrics["cost_usd"]["mean"],
            )
        )
    if pairwise_stats:
        lines.extend(["", "### Significance", ""])
        for comparison_name, stats in pairwise_stats.items():
            lines.append(f"- {comparison_name}: p={stats['p_value']:.6f}, statistic={stats['statistic']:.6f}")
    lines.append("")
    return "\n".join(lines)


def _write_field_chart(results_by_system: dict[str, list[dict[str, Any]]], output_dir: Path) -> Path | None:
    field_names = sorted(
        {
            field
            for records in results_by_system.values()
            for record in records
            for field in record["metrics"].get("field_f1", {})
        }
    )
    if not field_names:
        return None

    figure, axis = plt.subplots(figsize=(10, 4))
    for system_name, records in results_by_system.items():
        y_values = []
        for field_name in field_names:
            field_scores = [float(record["metrics"].get("field_f1", {}).get(field_name, 0.0)) for record in records]
            y_values.append(round(sum(field_scores) / len(field_scores), 6) if field_scores else 0.0)
        axis.plot(field_names, y_values, marker="o", label=system_name)

    axis.set_title("Per-Field F1 Comparison")
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    chart_path = output_dir / "per_field_f1.png"
    figure.savefig(chart_path)
    plt.close(figure)
    return chart_path


def _mean_and_std(metric_rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return _mean_and_std_from_values([float(row.get(key, 0.0)) for row in metric_rows])


def _mean_and_std_from_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": round(float(values[0]), 6), "std": 0.0}
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }
