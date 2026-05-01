from __future__ import annotations

from document_intelligence_engine.evaluation.evaluator import Evaluator
from document_intelligence_engine.evaluation.metrics import (
    compute_document_exact_match,
    compute_field_f1_scores,
    compute_field_level_f1,
    compute_hallucination_rate,
    compute_schema_validity,
)
from document_intelligence_engine.evaluation.stats import mcnemar_test


def test_experiment_metrics_exact_match_and_f1():
    prediction = {
        "invoice_number": "INV-1023",
        "date": "2025-01-12",
        "vendor": "ABC Corp",
        "total_amount": 1000.0,
        "line_items": [{"description": "Widget A", "quantity": 2.0, "unit_price": 500.0}],
    }
    ground_truth = dict(prediction)

    assert compute_document_exact_match(prediction, ground_truth) == 1.0
    assert compute_field_level_f1(prediction, ground_truth) == 1.0
    assert compute_field_f1_scores(prediction, ground_truth)["vendor"] == 1.0


def test_experiment_metrics_schema_and_hallucination():
    prediction = {
        "invoice_number": "INV-404",
        "date": "2025-01-12",
        "vendor": "Ghost Corp",
        "total_amount": 1000.0,
        "line_items": [],
    }
    schema = {
        "invoice_number": str,
        "date": str,
        "vendor": str,
        "total_amount": (int, float),
        "line_items": list,
    }

    assert compute_schema_validity(prediction, schema) == 1.0
    assert compute_hallucination_rate(prediction, "Invoice Number: INV-404\nDate: 2025-01-12\nTotal: 1000.00") > 0.0


def test_evaluator_returns_document_metrics():
    evaluator = Evaluator()
    output = {
        "extracted_fields": {
            "invoice_number": "INV-1023",
            "date": "2025-01-12",
            "vendor": "ABC Corp",
            "total_amount": 1000.0,
            "line_items": [],
        },
        "latency_ms": 12.5,
        "cost_usd": 0.001,
        "_errors": [],
        "_constraint_flags": [],
    }
    ground_truth = {
        "invoice_number": "INV-1023",
        "date": "2025-01-12",
        "vendor": "ABC Corp",
        "total_amount": 1000.0,
        "line_items": [],
    }

    result = evaluator.evaluate(output, ground_truth, source_text="Invoice Number INV-1023")

    assert result["exact_match"] == 1.0
    assert result["field_level_f1"] == 1.0
    assert result["schema_valid"] == 1.0
    assert result["latency_ms"] == 12.5


def test_mcnemar_test_counts_disagreements():
    result = mcnemar_test([True, True, False, False], [True, False, True, False])

    assert result["a_only"] == 1
    assert result["b_only"] == 1
    assert 0.0 <= float(result["p_value"]) <= 1.0
