from __future__ import annotations

from document_intelligence_engine.postprocessing.deterministic import apply_constraints
from document_intelligence_engine.postprocessing.normalizer import normalize_amount, normalize_date
from document_intelligence_engine.postprocessing.validator import validate_document


def test_apply_constraints_flags_sum_mismatch() -> None:
    result = apply_constraints(
        {
            "total_amount": 20.0,
            "line_items": [{"price": 9.0, "quantity": 2}],
        }
    )
    assert "line_items_sum_mismatch" in result.flags


def test_normalize_amount() -> None:
    assert normalize_amount("$1,024.55") == 1024.55


def test_normalize_date() -> None:
    assert normalize_date("2025-01-15") == "2025-01-15"


def test_validate_document_clears_invalid_invoice_number() -> None:
    validated = validate_document({"invoice_number": "***"})
    assert validated["invoice_number"] is None
