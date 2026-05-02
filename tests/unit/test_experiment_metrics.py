from __future__ import annotations

from document_intelligence_engine.evaluation.evaluator import Evaluator
from document_intelligence_engine.evaluation.metrics import (
    collect_hallucination_checks,
    compute_document_exact_match,
    compute_field_f1_scores,
    compute_field_level_f1,
    compute_hallucination_rate,
    compute_schema_validity,
)
from document_intelligence_engine.evaluation.schema import EXTRACTION_SCHEMA
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


def test_experiment_metrics_normalize_dates_amounts_and_line_item_order():
    prediction = {
        "invoice_number": "INV-1023",
        "date": "01/12/2025",
        "vendor": "ABC Corp",
        "total_amount": "$1000.00",
        "line_items": [
            {"description": "Widget B", "quantity": "1", "unit_price": "$50.00"},
            {"description": "Widget A", "quantity": 2.0, "unit_price": 500.0},
        ],
    }
    ground_truth = {
        "invoice_number": "INV-1023",
        "date": "2025-01-12",
        "vendor": "ABC Corp",
        "total_amount": 1000.0,
        "line_items": [
            {"description": "Widget A", "quantity": 2.0, "unit_price": 500.0},
            {"description": "Widget B", "quantity": 1.0, "unit_price": 50.0},
        ],
    }

    assert compute_document_exact_match(prediction, ground_truth) == 1.0


def test_experiment_metrics_normalize_text_case_whitespace_and_trailing_punctuation():
    prediction = {
        "invoice_number": " inv-1023 ",
        "date": "January 12, 2025",
        "vendor": "ABC Corp. ",
        "total_amount": "1,000.5",
        "line_items": [{"description": " Widget A ", "quantity": "2.0", "unit_price": "$500.00"}],
    }
    ground_truth = {
        "invoice_number": "INV-1023",
        "date": "2025-01-12",
        "vendor": "abc corp",
        "total_amount": "1000.50",
        "line_items": [{"description": "widget a", "quantity": 2, "unit_price": 500}],
    }

    assert compute_document_exact_match(prediction, ground_truth) == 1.0


def test_experiment_metrics_schema_and_hallucination():
    prediction = {
        "invoice_number": "INV-404",
        "date": "2025-01-12",
        "vendor": "Ghost Corp",
        "total_amount": 1000.0,
        "line_items": [],
    }
    assert compute_schema_validity(prediction, EXTRACTION_SCHEMA) == 1.0
    assert compute_hallucination_rate(prediction, "Invoice Number: INV-404\nDate: 2025-01-12\nTotal: 1000.00") > 0.0


def test_hallucination_normalizes_currency_and_dates_against_source_text():
    prediction = {
        "invoice_number": "INV-404",
        "date": "2025-01-12",
        "vendor": "ABC Corp.",
        "total_amount": "1,000.50",
        "line_items": [{"description": "Widget A", "quantity": 2, "unit_price": "$500.25"}],
    }
    source_text = "Invoice Number: INV-404\nDate: 01/12/2025\nVendor: ABC Corp\nTotal: $1,000.50\nWidget A 2 500.25"

    assert compute_hallucination_rate(prediction, source_text) == 0.0


def test_hallucination_tolerates_interleaved_numbers_and_minor_ocr_noise_in_text_fields():
    prediction = {
        "invoice_number": None,
        "date": None,
        "vendor": "Puyo",
        "total_amount": None,
        "line_items": [
            {"description": "FL-Xmas Off", "quantity": None, "unit_price": None},
            {"description": "Siky Lychee", "quantity": None, "unit_price": None},
        ],
    }
    source_text = (
        "Puyo 6 (Package) 1x 70,000 70,000 "
        "Vanilla SLC 1 FL-Xmas 30 Off 68,180 "
        "Si1ky Lychee 2x Sub Total 70,000"
    )

    assert compute_hallucination_rate(prediction, source_text) == 0.0


def test_hallucination_matches_month_name_dates_and_mixed_alnum_receipt_tokens():
    prediction = {
        "invoice_number": None,
        "date": "1995-03-17",
        "vendor": None,
        "total_amount": None,
        "line_items": [{"description": "DE.PSANG IJO MDM", "quantity": None, "unit_price": None}],
    }
    source_text = "MARCH 17, 1995 1 DE13.PSANG IJO MDM 29,800"

    assert compute_hallucination_rate(prediction, source_text) == 0.0


def test_collect_hallucination_checks_returns_field_paths_for_residual_flags():
    prediction = {
        "invoice_number": "INV-404",
        "date": None,
        "vendor": "Ghost Corp",
        "total_amount": None,
        "line_items": [{"description": "Imaginary Item", "quantity": None, "unit_price": None}],
    }
    checks = collect_hallucination_checks(prediction, "Invoice Number: INV-404")

    flagged_paths = {check["field_path"] for check in checks if check["counted"] and not check["grounded"]}
    assert "vendor" in flagged_paths
    assert "line_items[0].description" in flagged_paths


def test_experiment_metrics_schema_rejects_non_dict_line_items():
    prediction = {
        "invoice_number": "INV-404",
        "date": "2025-01-12",
        "vendor": "Ghost Corp",
        "total_amount": 1000.0,
        "line_items": ["bad-item"],
    }
    assert compute_schema_validity(prediction, EXTRACTION_SCHEMA) == 0.0


def test_experiment_metrics_schema_rejects_invalid_date_format():
    prediction = {
        "invoice_number": "INV-404",
        "date": "01/12/2025",
        "vendor": "Ghost Corp",
        "total_amount": 1000.0,
        "line_items": [],
    }

    assert compute_schema_validity(prediction, EXTRACTION_SCHEMA) == 0.0


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
