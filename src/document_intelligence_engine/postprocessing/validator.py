"""Field validation for structured output."""

from __future__ import annotations

import re


INVOICE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_-]{1,63}$")


def validate_document(payload: dict[str, object]) -> dict[str, object]:
    validated = dict(payload)
    invoice_number = validated.get("invoice_number")
    if invoice_number is not None and not INVOICE_PATTERN.match(str(invoice_number)):
        validated["invoice_number"] = None
    if validated.get("date") in {"", None}:
        validated["date"] = None
    total_amount = validated.get("total_amount")
    if total_amount is not None and not isinstance(total_amount, (int, float)):
        validated["total_amount"] = None
    return validated
