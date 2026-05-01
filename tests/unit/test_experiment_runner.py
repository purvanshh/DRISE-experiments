from __future__ import annotations

import json
from pathlib import Path

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.report import generate_experiment_report
from document_intelligence_engine.evaluation.runner import ExperimentRunner
from document_intelligence_engine.pipelines.base import BasePipeline
from run_experiments import _build_systems


class StaticPipeline(BasePipeline):
    def __init__(self, name: str, extracted_fields: dict[str, object]) -> None:
        self.name = name
        self.extracted_fields = extracted_fields

    def run(self, document):
        return {
            "extracted_fields": dict(self.extracted_fields),
            "confidences": {key: 1.0 for key in self.extracted_fields},
            "_constraint_flags": [],
            "_errors": [],
            "latency_ms": 1.0,
            "cost_usd": 0.0,
        }


def test_annotation_loader_resolves_relative_paths(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    dataset_path.write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "image_path": "image.png",
                "ocr_tokens": [{"text": "Invoice"}],
                "ground_truth": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_annotations(dataset_path)

    assert records[0]["doc_id"] == "doc-1"
    assert records[0]["image_path"] == str(image_path.resolve())
    assert records[0]["ocr_text"] == "Invoice"


def test_annotation_loader_normalizes_ground_truth_defaults(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    dataset_path.write_text(
        json.dumps(
            {
                "doc_id": "doc-2",
                "image_path": "image.png",
                "ground_truth": {"invoice_number": "INV-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_annotations(dataset_path)

    assert records[0]["ground_truth"] == {
        "invoice_number": "INV-1",
        "date": "",
        "vendor": "",
        "total_amount": None,
        "line_items": [],
    }


def test_experiment_runner_and_report_generation(tmp_path):
    dataset = [
        {
            "doc_id": "doc-1",
            "ocr_text": "Invoice Number: INV-1023",
            "ground_truth": {
                "invoice_number": "INV-1023",
                "date": None,
                "vendor": None,
                "total_amount": None,
                "line_items": [],
            },
        }
    ]
    runner = ExperimentRunner(
        [
            StaticPipeline(
                "system_a",
                {
                    "invoice_number": "INV-1023",
                    "date": None,
                    "vendor": None,
                    "total_amount": None,
                    "line_items": [],
                },
            ),
            StaticPipeline(
                "system_b",
                {
                    "invoice_number": "INV-0000",
                    "date": None,
                    "vendor": None,
                    "total_amount": None,
                    "line_items": [],
                },
            ),
        ],
        results_dir=tmp_path / "results",
    )

    results = runner.run(dataset)
    report = generate_experiment_report(results, output_dir=tmp_path / "results")

    assert (tmp_path / "results" / "system_a.json").exists()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "ablation_summary.csv").exists()
    assert report["summary"]["system_a"]["exact_match"]["mean"] == 1.0
    assert report["summary"]["system_b"]["exact_match"]["mean"] == 0.0


def test_experiment_runner_resumes_from_existing_records(tmp_path):
    dataset = [
        {
            "doc_id": "doc-1",
            "ocr_text": "Invoice Number: INV-1023",
            "ground_truth": {
                "invoice_number": "INV-1023",
                "date": None,
                "vendor": None,
                "total_amount": None,
                "line_items": [],
            },
        }
    ]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    existing_record = [
        {
            "doc_id": "doc-1",
            "metrics": {
                "field_f1": {"invoice_number": 1.0},
                "field_level_f1": 1.0,
                "exact_match": 1.0,
                "schema_valid": 1.0,
                "hallucination_rate": 0.0,
                "latency_ms": 0.5,
                "cost_usd": 0.0,
                "errors": [],
                "constraint_flags": [],
            },
            "output": {"cached": True},
        }
    ]
    (results_dir / "system_a.json").write_text(json.dumps(existing_record), encoding="utf-8")

    runner = ExperimentRunner(
        [StaticPipeline("system_a", {"invoice_number": "SHOULD_NOT_RUN", "line_items": []})],
        results_dir=results_dir,
        resume=True,
    )
    results = runner.run(dataset)

    assert results["system_a"][0]["output"] == {"cached": True}


def test_build_systems_includes_drise_ablations(monkeypatch):
    class StubPipeline:
        default_name = "stub"

        def __init__(self, config):
            self.name = config.get("name", self.default_name)

    class StubLLMOnlyPipeline(StubPipeline):
        default_name = "llm_only"

    class StubRAGPipeline(StubPipeline):
        default_name = "rag_llm"

    class StubDRISEPipeline(StubPipeline):
        default_name = "drise"

    monkeypatch.setattr("run_experiments.LLMOnlyPipeline", StubLLMOnlyPipeline)
    monkeypatch.setattr("run_experiments.RAGLLMPipeline", StubRAGPipeline)
    monkeypatch.setattr("run_experiments.DRISEPipeline", StubDRISEPipeline)

    systems = _build_systems(
        {
            "systems": ["llm_only", "rag_llm", "drise"],
            "llm": {"backend": "mock", "model": "mock-llm"},
            "retrieval": {"embedder": "hashing"},
            "drise": {"use_layout": True, "use_constraints": True},
            "ablation": {
                "enabled": True,
                "variants": {
                    "drise_no_layout": {"use_layout": False, "use_constraints": True},
                    "drise_no_constraints": {"use_layout": True, "use_constraints": False},
                },
            },
            "output_dir": str(Path("experiments/results")),
        }
    )

    names = [system.name for system in systems]
    assert names == ["llm_only", "rag_llm", "drise", "drise_no_layout", "drise_no_constraints"]


def test_report_generates_ablation_summary_for_drise_variants(tmp_path):
    report = generate_experiment_report(
        {
            "drise": [
                {
                    "doc_id": "doc-1",
                    "metrics": {
                        "field_f1": {"invoice_number": 1.0},
                        "field_level_f1": 1.0,
                        "exact_match": 1.0,
                        "schema_valid": 1.0,
                        "hallucination_rate": 0.0,
                        "latency_ms": 1.0,
                        "cost_usd": 0.0,
                    },
                }
            ],
            "drise_no_layout": [
                {
                    "doc_id": "doc-1",
                    "metrics": {
                        "field_f1": {"invoice_number": 0.5},
                        "field_level_f1": 0.5,
                        "exact_match": 0.0,
                        "schema_valid": 1.0,
                        "hallucination_rate": 0.2,
                        "latency_ms": 1.0,
                        "cost_usd": 0.0,
                    },
                }
            ],
        },
        output_dir=tmp_path / "results",
    )

    assert report["ablation_summary"][0]["system"] == "drise_no_layout"
    assert report["ablation_summary"][0]["delta_field_level_f1"] == -0.5
    assert (tmp_path / "results" / "ablation_summary.csv").exists()
