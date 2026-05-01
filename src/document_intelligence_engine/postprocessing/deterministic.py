"""Deterministic cross-field constraints."""

from __future__ import annotations

from math import isclose

from document_intelligence_engine.domain.contracts import ConstraintResult


def apply_constraints(payload: dict[str, object]) -> ConstraintResult:
    normalized = dict(payload)
    flags: list[str] = []
    line_items = normalized.get("line_items", [])
    total_amount = normalized.get("total_amount")

    if isinstance(line_items, list) and isinstance(total_amount, (int, float)):
        computed_total = 0.0
        for item in line_items:
            if not isinstance(item, dict):
                flags.append("invalid_line_item")
                continue
            price = item.get("price")
            quantity = item.get("quantity", 1)
            if isinstance(price, (int, float)) and isinstance(quantity, (int, float)):
                computed_total += float(price) * float(quantity)
        if not isclose(computed_total, float(total_amount), rel_tol=0.0, abs_tol=0.01):
            flags.append("line_items_sum_mismatch")

    normalized["_constraint_flags"] = flags
    return ConstraintResult(normalized=normalized, flags=flags)
