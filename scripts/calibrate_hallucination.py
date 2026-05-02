"""Generate a deterministic hallucination calibration sample and summary."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.metrics import collect_hallucination_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate hallucination checks against a deterministic sample.")
    parser.add_argument("--dataset", default="data/annotations/test.jsonl", help="Annotation dataset used for evaluation.")
    parser.add_argument("--results-file", default="experiments/results/drise.json", help="Per-system JSON results file.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of documents to sample for calibration.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--output-json",
        default="experiments/results/hallucination_calibration.json",
        help="Detailed calibration sample output.",
    )
    parser.add_argument(
        "--output-summary",
        default="experiments/results/hallucination_calibration_summary.json",
        help="Calibration summary output.",
    )
    args = parser.parse_args()

    dataset = load_annotations(args.dataset)
    dataset_by_id = {str(document["doc_id"]): document for document in dataset}
    results_path = (PROJECT_ROOT / args.results_file).resolve()
    output_json_path = (PROJECT_ROOT / args.output_json).resolve()
    output_summary_path = (PROJECT_ROOT / args.output_summary).resolve()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    sampled_records = _sample_records(results, args.sample_size, args.seed)

    calibration_rows: list[dict[str, Any]] = []
    for record in sampled_records:
        doc_id = str(record["doc_id"])
        dataset_record = dataset_by_id.get(doc_id)
        if dataset_record is None:
            continue
        checks = collect_hallucination_checks(record["output"]["extracted_fields"], str(dataset_record.get("ocr_text", "")))
        flagged_checks = [check for check in checks if check["counted"] and not check["grounded"]]
        if not flagged_checks:
            continue
        calibration_rows.append(
            {
                "doc_id": doc_id,
                "ocr_text": str(dataset_record.get("ocr_text", "")),
                "checked_value_count": sum(1 for check in checks if check["counted"]),
                "flagged_value_count": len(flagged_checks),
                "flagged_checks": flagged_checks,
            }
        )

    sample_checked = sum(row["checked_value_count"] for row in calibration_rows)
    sample_flagged = sum(row["flagged_value_count"] for row in calibration_rows)
    full_checked, full_flagged = _aggregate_counts(results, dataset_by_id)
    sample_macro_rate = _mean_metric(sampled_records, "hallucination_rate")
    full_macro_rate = _mean_metric(results, "hallucination_rate")
    summary = {
        "sample_doc_count": len(sampled_records),
        "sample_docs_with_flags": len(calibration_rows),
        "sample_checked_values": sample_checked,
        "sample_flagged_values": sample_flagged,
        "automatic_sample_hallucination_rate": round(sample_flagged / sample_checked, 6) if sample_checked else 0.0,
        "automatic_sample_document_mean_rate": round(sample_macro_rate, 6),
        "full_checked_values": full_checked,
        "full_flagged_values": full_flagged,
        "automatic_full_hallucination_rate": round(full_flagged / full_checked, 6) if full_checked else 0.0,
        "automatic_full_document_mean_rate": round(full_macro_rate, 6),
        "calibration_notes": [
            "Sample uses deterministic random seed 42 over stored DRISE outputs.",
            "Only fields still flagged after normalized grounding checks are emitted for review.",
            "This sample is intended for manual inspection of residual false positives rather than raw substring mismatches.",
        ],
    }

    output_json_path.write_text(json.dumps(calibration_rows, indent=2), encoding="utf-8")
    output_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _sample_records(results: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(results):
        return list(results)
    randomizer = random.Random(seed)
    return randomizer.sample(results, sample_size)


def _aggregate_counts(
    results: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    checked = 0
    flagged = 0
    for record in results:
        doc_id = str(record["doc_id"])
        dataset_record = dataset_by_id.get(doc_id)
        if dataset_record is None:
            continue
        checks = collect_hallucination_checks(record["output"]["extracted_fields"], str(dataset_record.get("ocr_text", "")))
        checked += sum(1 for check in checks if check["counted"])
        flagged += sum(1 for check in checks if check["counted"] and not check["grounded"])
    return checked, flagged


def _mean_metric(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [float(record.get("metrics", {}).get(metric_name, 0.0)) for record in results]
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
