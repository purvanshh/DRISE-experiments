from __future__ import annotations

import json

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.report import generate_experiment_report
from document_intelligence_engine.evaluation.runner import ExperimentRunner
from document_intelligence_engine.pipelines.base import BasePipeline


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
    assert report["summary"]["system_a"]["exact_match"]["mean"] == 1.0
    assert report["summary"]["system_b"]["exact_match"]["mean"] == 0.0
