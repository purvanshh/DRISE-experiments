from __future__ import annotations

from datetime import date, timedelta

from postprocessing.confidence import apply_confidence_policy
from postprocessing.constraints import apply_constraints
from postprocessing.validation import validate_fields


def test_validation_rules(settings):
    document, errors = validate_fields(
        [
            {"field": "invoice_number", "value": "***", "confidence": 0.91, "key": "Invoice Number"},
            {"field": "date", "value": "2025-13-40", "confidence": 0.92, "key": "Date"},
            {"field": "total_amount", "value": "not-a-number", "confidence": 0.93, "key": "Total Amount"},
        ],
        settings,
    )

    error_codes = {error["code"] for error in errors}

    assert document["invoice_number"]["valid"] is False
    assert document["date"]["valid"] is False
    assert document["total_amount"]["valid"] is False
    assert {"regex_validation_failed", "invalid_date", "invalid_numeric"} <= error_codes


def test_constraint_logic(settings):
    future_date = (date.today() + timedelta(days=1)).isoformat()

    document, errors, flags = apply_constraints(
        {
            "date": {"value": future_date, "confidence": 0.9, "valid": True},
            "total_amount": {"value": 10.0, "confidence": 0.9, "valid": True},
            "line_items": {"value": [{"price": 3.0, "quantity": 2}], "confidence": 0.9, "valid": True},
        },
        settings,
    )

    error_codes = {error["code"] for error in errors}

    assert document["invoice_number"]["valid"] is False
    assert "missing_required_field:invoice_number" in flags
    assert "future_date:date" in flags
    assert "line_items_total_mismatch" in flags
    assert {"missing_required_field", "future_date", "line_items_total_mismatch"} <= error_codes


def test_confidence_threshold_policy(settings):
    filtered_document, errors = apply_confidence_policy(
        {
            "invoice_number": {"value": "INV-1023", "confidence": 0.95, "valid": True},
            "total_amount": {"value": 1200.5, "confidence": 0.2, "valid": True},
        },
        settings,
    )

    assert "invoice_number" in filtered_document
    assert "total_amount" not in filtered_document
    assert errors == [
        {
            "field": "total_amount",
            "code": "low_confidence",
            "message": "Field confidence 0.200000 is below threshold 0.600000.",
        }
    ]
