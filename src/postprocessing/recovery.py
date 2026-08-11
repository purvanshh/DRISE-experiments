"""Heuristic field recovery from OCR tokens when model spans are incomplete."""

from __future__ import annotations

import re
from statistics import fmean
from typing import Any


DATE_PATTERNS = (
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", re.IGNORECASE),
)
AMOUNT_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?(?!\d)")
# A numeric token that may use locale thousands separators (comma or dot)
# and either a comma or dot decimal separator (e.g. "28,000", "28.000",
# "1,000.50", "9.500,00", "1200.50").
_NUMERIC_TOKEN = re.compile(r"^[\d.,]+$")

# Words that delimit the totals section of a receipt. Lines containing these
# describe subtotals, taxes, tips, cash/change, or grand totals -- not items.
_SECTION_PATTERNS = (
    re.compile(r"\btotal\w*\b|\bsub\s*total\b", re.IGNORECASE),
    re.compile(r"\bsub\w*\b", re.IGNORECASE),
    re.compile(r"\bgrand\b|\bamount\s*due\b|\bbalance\s*due\b", re.IGNORECASE),
    re.compile(r"\bcash\b|\bchang\w*\b|\bkembal\w*\b|\bbayar\w*\b|\bpembayaran\b|\btunai\b", re.IGNORECASE),
    re.compile(r"\bcg\b|\btl\b|\brp\b|\bidr\b", re.IGNORECASE),
    re.compile(r"\bdisc\w*\b|\bdiskon\b", re.IGNORECASE),
    re.compile(r"\bpajak\b|\bpaj\b|\bppn\b|\bpb1\b|\bservi\w*\b|\bservice\s*charge\b", re.IGNORECASE),
    re.compile(r"\btax\b|\btips?\b|\binc\b|\bpayment\b|\bpay\b", re.IGNORECASE),
)


def recover_missing_entities(
    entities: list[dict[str, Any]],
    ocr_tokens: list[dict[str, Any]] | None,
    field_aliases: dict[str, str],
) -> list[dict[str, Any]]:
    if not ocr_tokens:
        return entities

    recovered_entities = list(entities)
    existing_by_field: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        existing_by_field.setdefault(str(entity.get("field")), []).append(entity)
    token_lines = _group_token_lines(ocr_tokens)
    full_text = "\n".join(line["text"] for line in token_lines)

    invoice_number = _recover_invoice_number(token_lines, full_text)
    if invoice_number and _should_recover_field(existing_by_field, "invoice_number", invoice_number):
        recovered_entities.append(_entity("invoice_number", invoice_number, "heuristic_invoice_number", token_lines))

    date_value = _recover_date(token_lines, full_text)
    if date_value and _should_recover_field(existing_by_field, "date", date_value):
        recovered_entities.append(_entity("date", date_value, "heuristic_date", token_lines))

    vendor = _recover_vendor(token_lines, field_aliases)
    if vendor and _should_recover_field(existing_by_field, "vendor", vendor):
        recovered_entities.append(_entity("vendor", vendor, "heuristic_vendor", token_lines))

    total_amount, total_confidence = _recover_total_amount(token_lines)
    if total_amount and _should_recover_field(existing_by_field, "total_amount", total_amount):
        recovered_entities.append(
            {
                "field": "total_amount",
                "key": "heuristic_total_amount",
                "value": total_amount,
                "confidence": round(total_confidence, 6),
                "source": "ocr_recovery",
            }
        )
    if _should_recover_line_items(existing_by_field) and _is_receipt_document(token_lines):
        line_items = _recover_line_items(token_lines)
        if line_items:
            recovered_entities.append(
                {
                    "field": "line_items",
                    "key": "heuristic_line_items",
                    "value": line_items,
                    "confidence": _line_items_confidence(line_items),
                    "source": "ocr_recovery",
                }
            )

    return recovered_entities


def _line_items_confidence(line_items: list[dict[str, Any]]) -> float:
    """Calibrate line-item confidence from structural completeness.

    Items with both a price and a quantity are scored higher than items that
    only carry a description and price, which are in turn higher than a bare
    description-only list.
    """
    if not line_items:
        return 0.5
    scores = []
    for item in line_items:
        price = item.get("unit_price", item.get("price"))
        quantity = item.get("quantity")
        if isinstance(price, (int, float)):
            if isinstance(quantity, (int, float)):
                scores.append(0.78)
            else:
                scores.append(0.68)
        else:
            scores.append(0.55)
    return round(max(0.5, min(0.78, fmean(scores))), 6)


def _recover_invoice_number(token_lines: list[dict[str, Any]], full_text: str) -> str | None:
    patterns = (
        re.compile(r"(?:invoice|receipt|bill)\s*(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/]*)", re.IGNORECASE),
        re.compile(r"\b(?:inv|rcpt)[\s:#-]*([A-Z0-9][A-Z0-9\-_/]*)\b", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(full_text)
        if match:
            return match.group(1).strip()
    return None


def _recover_date(token_lines: list[dict[str, Any]], full_text: str) -> str | None:
    prioritized = list(token_lines)
    prioritized.sort(key=lambda line: (0 if "date" in line["text"].lower() else 1, line["y"]))
    for line in prioritized:
        for pattern in DATE_PATTERNS:
            match = pattern.search(line["text"])
            if match:
                return match.group(0)
    for pattern in DATE_PATTERNS:
        match = pattern.search(full_text)
        if match:
            return match.group(0)
    return None


def _recover_vendor(token_lines: list[dict[str, Any]], field_aliases: dict[str, str]) -> str | None:
    alias_tokens = {alias.lower() for alias in field_aliases}
    for line in token_lines[:5]:
        text = line["text"].strip()
        lowered = text.lower()
        if not text or any(token in lowered for token in ("total", "tax", "subtotal", "invoice", "receipt", "date", "cash", "change")):
            continue
        if lowered in alias_tokens:
            continue
        digit_ratio = sum(character.isdigit() for character in text) / max(len(text), 1)
        if digit_ratio > 0.05:
            continue
        if not re.search(r"\b(?:store|vendor|merchant|seller|supplier|corp|inc|llc|ltd|cafe|restaurant)\b", lowered):
            continue
        if len(text.split()) >= 2 or text.isupper():
            return text
    return None


def _recover_total_amount(token_lines: list[dict[str, Any]]) -> tuple[str | None, float]:
    """Recover the grand total amount and a calibrated confidence.

    Priority is GRAND TOTAL > TOTAL/AMOUNT DUE/TL > SUBTOTAL. Cash tendered,
    change, and tax lines are never selected unless no total line exists.
    Returns ``(value, confidence)``; confidence reflects how strongly the
    amount is anchored to a total keyword.
    """
    scored_candidates: list[tuple[float, float, str, float]] = []
    for index, line in enumerate(token_lines):
        text = _merge_space_thousands(line["text"].strip())
        lowered = text.lower()
        # "Total Item: 1" / "Total Qty: 3" are item counts, not money.
        if re.search(r"\btotal\s*(item|qty|count|barang)\b", lowered):
            continue
        amounts = _token_amounts(text)
        if not amounts:
            continue
        if re.search(r"\b(grand\s+total|total\s+due|amount\s+due|balance\s+due)\b", lowered):
            score = 6.0
            confidence = 0.78
        elif re.search(r"\btotal\b|\btl\b|\bnet\b", lowered):
            score = 4.0
            confidence = 0.72
        elif "subtotal" in lowered or re.search(r"\bsub\b", lowered):
            score = 1.0
            confidence = 0.55
        else:
            continue
        # Prefer later lines: grand total usually appears below subtotal/tax.
        score += 0.2 * index
        scored_candidates.append((score, index, amounts[-1][1], confidence))
    if scored_candidates:
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return _format_amount_return(scored_candidates[0][2]), scored_candidates[0][3]

    # Fallback: the last plausible amount in the document. Huge numbers (card
    # numbers, phone numbers) and zero are filtered out.
    last_amount: float | None = None
    for line in token_lines:
        for _, value in _token_amounts(_merge_space_thousands(line["text"])):
            if 1 <= value <= 50_000_000:
                last_amount = value
    if last_amount is None:
        return None, 0.0
    return _format_amount_return(last_amount), 0.5


def _format_amount_return(value: float) -> str:
    """Serialize a recovered amount back to a canonical string for normalization."""
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def _merge_space_thousands(text: str) -> str:
    """Merge space-separated thousands groups used by some locales.

    ``"154 000"`` -> ``"154000"``, ``"18 000"`` -> ``"18000"`` while leaving
    quantities next to prices untouched (``"1 28,000"`` stays unchanged).
    """
    return re.sub(r"(?<=\d) (?=\d{3}(?:\s|$))", "", text)


def _token_amounts(text: str) -> list[tuple[str, float]]:
    """Return ``(raw_token, float_value)`` pairs for numeric tokens in text.

    Only whitespace-delimited tokens that are purely numeric characters are
    considered, so alphanumeric tokens like ``RB0006`` or ``1.5L`` are ignored.
    """
    pairs: list[tuple[str, float]] = []
    for token in text.split():
        if not _NUMERIC_TOKEN.match(token):
            continue
        value = _parse_amount_token(token)
        if value is not None:
            pairs.append((token, value))
    return pairs


def _parse_amount_token(token: str) -> float | None:
    """Parse a single numeric token with locale separator handling."""
    text = token
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            # "9.500,00" -> 9500.0 (dot thousands, comma decimal)
            text = text.replace(".", "").replace(",", ".")
        else:
            # "1,000.50" -> 1000.5 (comma thousands, dot decimal)
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[0]) in (1, 2) and len(parts[1]) in (1, 2):
            # "73,45" -> 73.45 (comma decimal)
            text = text.replace(",", ".")
        else:
            # "28,000" -> 28000 (comma thousands)
            text = text.replace(",", "")
    elif "." in text:
        parts = text.split(".")
        if len(parts) == 2 and len(parts[0]) <= 3 and parts[1].isdigit() and len(parts[1]) == 3:
            # "18.000" / "28.000" -> thousands separator
            text = text.replace(".", "")
        elif len(parts) > 2:
            # "1.234.567" -> thousands separators
            text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def _is_quantity(value: Any) -> bool:
    """Return True when a numeric value looks like a line-item count (1-99)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return float(value).is_integer() and 1 <= float(value) <= 99


def _recover_line_items(token_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover line items from OCR text lines.

    Supports the common receipt layouts:
      - ``description quantity unit_price``
      - ``description unit_price quantity line_total``
      - ``description unit_price``
      - ``quantity description unit_price``
      - ``description`` followed by a ``quantity [x] unit_price`` line

    Quantities may use ``x`` syntax (``2X 24,000``). Locale thousands and
    decimal separators are handled for both comma and dot based locales.
    """
    line_items: list[dict[str, Any]] = []
    pending_description: str | None = None
    for line in token_lines:
        text = line["text"].strip()
        if not text:
            continue
        normalized = _merge_space_thousands(text)
        if _is_section_line(normalized):
            pending_description = None
            continue

        item = _parse_item_line(normalized)
        if item is None:
            if pending_description and _token_amounts(normalized):
                numbers = [value for _, value in _token_amounts(normalized)]
                quantity, unit_price = _resolve_quantity_price(numbers, None)
                if unit_price is not None and pending_description:
                    line_items.append(
                        {
                            "description": pending_description,
                            "quantity": _to_native_number(quantity) if quantity is not None else None,
                            "unit_price": _to_native_number(unit_price),
                        }
                    )
                pending_description = None
            elif _looks_like_description_only(normalized):
                cleaned = _clean_description(normalized)
                pending_description = f"{pending_description} {cleaned}".strip() if pending_description else cleaned
            continue

        if pending_description and not item.get("description"):
            item["description"] = pending_description
            pending_description = None
        elif pending_description:
            pending_description = None

        if item.get("description"):
            line_items.append(item)
    return line_items


def _is_section_line(text: str) -> bool:
    """Return True when a line belongs to the totals/tax/cash section."""
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in _SECTION_PATTERNS)


def _is_receipt_document(token_lines: list[dict[str, Any]]) -> bool:
    """Return True when the OCR resembles a receipt rather than a form.

    Receipts contain totals-section keywords and several numeric amount tokens;
    forms (FUNSD) generally do not. Used to avoid fabricating line items on
    documents that have none.
    """
    if not token_lines:
        return False
    text = " ".join(line["text"] for line in token_lines).lower()
    keyword_hits = sum(1 for pattern in _SECTION_PATTERNS if pattern.search(text))
    amount_hits = sum(len(_token_amounts(_merge_space_thousands(line["text"]))) for line in token_lines)
    return keyword_hits >= 1 and amount_hits >= 2


def _parse_item_line(text: str) -> dict[str, Any] | None:
    """Parse a single line into a line item or return None.

    A line is a candidate item when it has at least one description token and
    at least one numeric token, or when it is a pure numeric continuation line
    (handled by the caller attaching a pending description).
    """
    tokens = text.split()
    if not tokens:
        return None

    # Leading quantity: "2 Mineral Water 18 000" -> qty 2; "2X 24,000" -> qty 2.
    lead_quantity: int | None = None
    start = 0
    first = tokens[0]
    lead_match = re.match(r"^(\d{1,2})\s*(?:x)?$", first, flags=re.IGNORECASE)
    if lead_match:
        lead_quantity = int(lead_match.group(1))
        start = 1

    description_parts: list[str] = []
    number_tokens: list[str] = []
    for token in tokens[start:]:
        if _NUMERIC_TOKEN.match(token):
            number_tokens.append(token)
        elif token.lower() not in {"x", "@"}:
            description_parts.append(token)

    if not number_tokens:
        return None

    numbers = [value for _, value in (_token_amounts(" ".join(number_tokens)))]

    description = _clean_description(" ".join(description_parts))
    if not description and not lead_quantity:
        return None

    quantity = float(lead_quantity) if lead_quantity is not None else None
    quantity, unit_price = _resolve_quantity_price(numbers, quantity)

    return {
        "description": description,
        "quantity": _to_native_number(quantity) if quantity is not None else None,
        "unit_price": _to_native_number(unit_price) if unit_price is not None else None,
    }


def _resolve_quantity_price(numbers: list[float], quantity: float | None) -> tuple[float | None, float | None]:
    """Determine (quantity, unit_price) from the numeric tokens of a line."""
    if not numbers:
        return quantity, None
    if max(numbers) > 1_000_000_000:
        return quantity, None
    if len(numbers) == 1:
        return quantity, numbers[0]
    if len(numbers) == 2:
        if quantity is not None:
            # Leading quantity + [unit_price, line_total] -> unit_price is the
            # first number that is not itself a quantity (e.g. "1 1 36,000").
            return quantity, _first_price(numbers)
        if _is_quantity(numbers[0]) and not _is_quantity(numbers[1]):
            return numbers[0], numbers[1]
        if _is_quantity(numbers[1]) and not _is_quantity(numbers[0]):
            return numbers[1], numbers[0]
        return quantity, numbers[-1]
    if quantity is not None:
        return quantity, _first_price(numbers)
    if _is_quantity(numbers[0]) and not _is_quantity(numbers[1]):
        return numbers[0], numbers[1]
    if _is_quantity(numbers[1]) and not _is_quantity(numbers[0]):
        return numbers[1], numbers[0]
    return quantity, numbers[0]


def _first_price(numbers: list[float]) -> float:
    """First number that is not quantity-like; falls back to the first number."""
    for number in numbers:
        if not _is_quantity(number):
            return number
    return numbers[0]


def _to_native_number(value: Any) -> int | float:
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def _group_token_lines(ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ocr_tokens:
        return []
    ordered_tokens = sorted(
        ocr_tokens,
        key=lambda token: (
            int(token.get("page_number", 1)),
            int(token.get("bbox", [0, 0, 0, 0])[1]),
            int(token.get("bbox", [0, 0, 0, 0])[0]),
        ),
    )
    lines: list[dict[str, Any]] = []
    current_tokens: list[dict[str, Any]] = []
    current_y: int | None = None
    for token in ordered_tokens:
        bbox = token.get("bbox", [0, 0, 0, 0])
        y0 = int(bbox[1]) if isinstance(bbox, list) and len(bbox) == 4 else 0
        if current_y is None or abs(y0 - current_y) <= 18:
            current_tokens.append(token)
            current_y = y0 if current_y is None else min(current_y, y0)
            continue
        lines.append(_finalize_line(current_tokens))
        current_tokens = [token]
        current_y = y0
    if current_tokens:
        lines.append(_finalize_line(current_tokens))
    return lines


def _finalize_line(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(tokens, key=lambda token: int(token.get("bbox", [0, 0, 0, 0])[0]))
    text = " ".join(str(token.get("text", "")).strip() for token in ordered if token.get("text")).strip()
    y_value = int(ordered[0].get("bbox", [0, 0, 0, 0])[1]) if ordered else 0
    return {
        "text": text,
        "y": y_value,
        "confidence": _line_confidence(ordered),
    }


def _entity(field: str, value: Any, key: str, token_lines: list[dict[str, Any]]) -> dict[str, Any]:
    confidence = _line_confidence(token_lines[:3]) if token_lines else 0.55
    return {
        "field": field,
        "key": key,
        "value": value,
        "confidence": confidence,
        "source": "ocr_recovery",
    }


def _line_confidence(tokens_or_lines: list[dict[str, Any]]) -> float:
    confidences = [float(item.get("confidence", 0.0)) for item in tokens_or_lines if isinstance(item.get("confidence"), (int, float))]
    if not confidences:
        return 0.55
    return round(max(0.45, min(0.75, fmean(confidences))), 6)


def _parse_number(value: str) -> float | int | None:
    cleaned = value.replace(",", "").strip()
    try:
        numeric = float(cleaned)
    except ValueError:
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _should_recover_field(existing_by_field: dict[str, list[dict[str, Any]]], field: str, recovered_value: Any) -> bool:
    existing = existing_by_field.get(field, [])
    if not existing:
        return True
    return not any(_looks_like_valid_field_value(field, entity.get("value")) for entity in existing)


def _should_recover_line_items(existing_by_field: dict[str, list[dict[str, Any]]]) -> bool:
    existing = existing_by_field.get("line_items", [])
    if not existing:
        return True
    for entity in existing:
        value = entity.get("value")
        if isinstance(value, list) and value:
            return False
    return True


def _looks_like_valid_field_value(field: str, value: Any) -> bool:
    if value in (None, "", []):
        return False
    text = str(value).strip()
    if field == "date":
        return any(pattern.search(text) for pattern in DATE_PATTERNS)
    if field == "invoice_number":
        return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9\-_/]*", text, flags=re.IGNORECASE))
    if field == "total_amount":
        return bool(_token_amounts(_merge_space_thousands(text)))
    if field == "vendor":
        return len(text.split()) >= 2 and not any(character.isdigit() for character in text)
    return True


def _looks_like_description_only(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("subtotal", "total", "tax", "cash", "change", "date", "invoice")):
        return False
    return bool(re.search(r"[A-Za-z]", text)) and not _token_amounts(text)


def _clean_description(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower().startswith("x "):
        text = text[2:].strip()
    return text.strip(" -")
