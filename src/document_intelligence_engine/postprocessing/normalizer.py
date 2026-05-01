"""Normalization helpers."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation


def normalize_string(value: object) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


def normalize_amount(value: object) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned:
        return None
    try:
        return float(Decimal(cleaned))
    except InvalidOperation:
        return None


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    candidates = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y")
    raw = str(value).strip()
    for fmt in candidates:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_document(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    if "invoice_number" in normalized:
        normalized["invoice_number"] = normalize_string(normalized["invoice_number"])
    if "vendor" in normalized:
        normalized["vendor"] = normalize_string(normalized["vendor"])
    if "date" in normalized:
        normalized["date"] = normalize_date(normalized["date"])
    if "total_amount" in normalized:
        normalized["total_amount"] = normalize_amount(normalized["total_amount"])
    return normalized
