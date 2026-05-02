"""Recompute experiment metrics from stored outputs without rerunning extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.evaluator import Evaluator
from document_intelligence_engine.evaluation.report import generate_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics for stored experiment outputs.")
    parser.add_argument("--dataset", default="data/annotations/test.jsonl", help="Annotation dataset used for evaluation.")
    parser.add_argument("--results-dir", default="experiments/results", help="Directory containing per-system JSON results.")
    args = parser.parse_args()

    dataset = load_annotations(args.dataset)
    dataset_by_id = {str(document["doc_id"]): document for document in dataset}
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    evaluator = Evaluator()
    results_by_system: dict[str, list[dict[str, Any]]] = {}

    for result_path in sorted(results_dir.glob("*.json")):
        if result_path.name in {"report.json", "summary.json", "pairwise_stats.json", "ablation_summary.json"}:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        if payload and not isinstance(payload[0], dict):
            continue
        if payload and not {"doc_id", "metrics", "output"}.issubset(payload[0]):
            continue
        recomputed_records: list[dict[str, Any]] = []
        for record in payload:
            doc_id = str(record["doc_id"])
            document = dataset_by_id.get(doc_id)
            if document is None:
                recomputed_records.append(record)
                continue
            output = record.get("output")
            if not isinstance(output, dict):
                recomputed_records.append(record)
                continue
            recomputed_metrics = evaluator.evaluate(
                output,
                document.get("ground_truth"),
                source_text=str(document.get("ocr_text", "")),
            )
            recomputed_records.append({**record, "metrics": recomputed_metrics})
        result_path.write_text(json.dumps(recomputed_records, indent=2), encoding="utf-8")
        results_by_system[result_path.stem] = recomputed_records

    report = generate_experiment_report(results_by_system, output_dir=results_dir)
    exact_match_summary = {
        system_name: metrics["exact_match"]["mean"] for system_name, metrics in report["summary"].items()
    }
    print(json.dumps(exact_match_summary, indent=2))


if __name__ == "__main__":
    main()
