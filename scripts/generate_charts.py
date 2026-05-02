"""Generate benchmark comparison charts from summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY_CSV = "experiments/results/summary.csv"
DEFAULT_SUMMARY_JSON = "experiments/results/summary.json"
DEFAULT_OUTPUT_DIR = "experiments/results"

METRIC_COLUMNS = {
    "Field F1": "field_level_f1_mean",
    "Exact Match": "exact_match_mean",
    "Schema Valid": "schema_valid_mean",
    "Hallucination": "hallucination_rate_mean",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts.")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Path to summary.csv")
    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help="Path to summary.json used for per-field F1 breakdowns",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write chart PNGs")
    args = parser.parse_args()

    summary_rows = _load_summary_rows(Path(args.summary_csv))
    summary_json = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_overall_metrics_chart(summary_rows, output_dir / "system_metrics.png")
    _write_per_field_chart(summary_json, output_dir / "per_field_f1_systems.png")


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_pointer:
        reader = csv.DictReader(file_pointer)
        return list(reader)


def _write_overall_metrics_chart(rows: list[dict[str, str]], path: Path) -> None:
    systems = [row["system"] for row in rows]
    metric_names = list(METRIC_COLUMNS)
    values = [[float(row[METRIC_COLUMNS[metric_name]]) for row in rows] for metric_name in metric_names]

    figure, axis = plt.subplots(figsize=(11, 5))
    x_positions = np.arange(len(systems))
    bar_width = 0.2
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, num=len(metric_names))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for index, metric_name in enumerate(metric_names):
        axis.bar(
            x_positions + offsets[index],
            values[index],
            width=bar_width,
            label=metric_name,
            color=colors[index],
        )

    axis.set_title("System-Level Benchmark Metrics")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(systems, rotation=25, ha="right")
    axis.legend(ncols=2)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _write_per_field_chart(summary_json: dict[str, dict], path: Path) -> None:
    systems = list(summary_json)
    field_names = sorted(
        {
            field_name
            for metrics in summary_json.values()
            for field_name in metrics.get("field_f1_breakdown", {})
        }
    )
    if not field_names:
        return

    figure, axis = plt.subplots(figsize=(11, 5))
    x_positions = np.arange(len(field_names))
    bar_width = min(0.14, 0.8 / max(len(systems), 1))
    offsets = np.linspace(
        -bar_width * (len(systems) - 1) / 2.0,
        bar_width * (len(systems) - 1) / 2.0,
        num=len(systems),
    )

    for index, system_name in enumerate(systems):
        field_breakdown = summary_json[system_name].get("field_f1_breakdown", {})
        values = [float(field_breakdown.get(field_name, {}).get("mean", 0.0) or 0.0) for field_name in field_names]
        axis.bar(x_positions + offsets[index], values, width=bar_width, label=system_name)

    axis.set_title("Per-Field F1 by System")
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(field_names, rotation=25, ha="right")
    axis.legend(ncols=2)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
