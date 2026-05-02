"""Experiment runner for document extraction systems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from document_intelligence_engine.domain.experiment_models import PipelineBase, ProcessedDocument
from document_intelligence_engine.evaluation.evaluator import Evaluator


class ExperimentRunner:
    """Run a set of extraction systems across a fixed evaluation dataset."""

    def __init__(
        self,
        systems: list[PipelineBase],
        *,
        evaluator: Evaluator | None = None,
        results_dir: str | Path = "experiments/results",
        resume: bool = True,
        max_total_cost_usd: float | None = None,
    ) -> None:
        self.systems = systems
        self.evaluator = evaluator or Evaluator()
        self.results_dir = Path(results_dir).expanduser().resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.max_total_cost_usd = max_total_cost_usd

    def run(self, dataset: list[ProcessedDocument]) -> dict[str, list[dict[str, Any]]]:
        all_results: dict[str, list[dict[str, Any]]] = {}
        cumulative_cost = 0.0
        for system in self.systems:
            output_path = self.results_dir / f"{system.name}.json"
            existing_records = self._load_existing_records(output_path) if self.resume else {}
            records_by_doc_id = dict(existing_records)
            cumulative_cost += self._summed_cost(existing_records.values())

            for document in dataset:
                doc_id = str(document["doc_id"])
                if doc_id in records_by_doc_id:
                    continue
                if self.max_total_cost_usd is not None and cumulative_cost >= self.max_total_cost_usd:
                    raise RuntimeError(
                        f"Experiment cost cap exceeded: ${cumulative_cost:.6f} >= ${self.max_total_cost_usd:.6f}"
                    )

                try:
                    output = system.run(document)
                    metrics = self.evaluator.evaluate(
                        output,
                        ground_truth=document.get("ground_truth"),
                        source_text=str(document.get("ocr_text", "")),
                    )
                    record = {
                        "doc_id": doc_id,
                        "metrics": metrics,
                        "output": output,
                    }
                except Exception as exc:
                    record = {
                        "doc_id": doc_id,
                        "metrics": {
                            "field_f1": {},
                            "field_level_f1": 0.0,
                            "exact_match": 0.0,
                            "schema_valid": 0.0,
                            "hallucination_rate": 0.0,
                            "latency_ms": 0.0,
                            "cost_usd": 0.0,
                            "errors": [str(exc)],
                            "constraint_flags": [],
                        },
                        "output": None,
                        "error": str(exc),
                    }
                records_by_doc_id[doc_id] = record
                cumulative_cost += float(record["metrics"].get("cost_usd", 0.0) or 0.0)
                self._write_records(output_path, records_by_doc_id)

            ordered_records = [records_by_doc_id[str(document["doc_id"])] for document in dataset if str(document["doc_id"]) in records_by_doc_id]
            all_results[system.name] = ordered_records
        return all_results

    @staticmethod
    def _load_existing_records(output_path: Path) -> dict[str, dict[str, Any]]:
        if not output_path.exists():
            return {}
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {str(record["doc_id"]): record for record in payload}

    @staticmethod
    def _write_records(output_path: Path, records_by_doc_id: dict[str, dict[str, Any]]) -> None:
        ordered_records = [records_by_doc_id[key] for key in sorted(records_by_doc_id)]
        output_path.write_text(json.dumps(ordered_records, indent=2), encoding="utf-8")

    @staticmethod
    def _summed_cost(records: Any) -> float:
        return sum(float(record.get("metrics", {}).get("cost_usd", 0.0) or 0.0) for record in records)
