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
LINE_ITEM_PATTERN = re.compile(
    r"^(?P<description>[A-Za-z][A-Za-z0-9 &()'.,/-]{2,}?)\s+(?P<quantity>\d+(?:\.\d+)?)\s+(?P<unit_price>\d{1,3}(?:,\d{3})*(?:\.\d{2})?)$"
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

    total_amount = _recover_total_amount(token_lines)
    if total_amount and _should_recover_field(existing_by_field, "total_amount", total_amount):
        recovered_entities.append(_entity("total_amount", total_amount, "heuristic_total_amount", token_lines))

    if _should_recover_line_items(existing_by_field):
        line_items = _recover_line_items(token_lines)
        if line_items:
            recovered_entities.append(
                {
                    "field": "line_items",
                    "key": "heuristic_line_items",
                    "value": line_items,
                    "confidence": _line_confidence(token_lines[: min(len(line_items), len(token_lines))]),
                    "source": "ocr_recovery",
                }
            )

    return recovered_entities


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


def _recover_total_amount(token_lines: list[dict[str, Any]]) -> str | None:
    scored_candidates: list[tuple[float, str]] = []
    for index, line in enumerate(token_lines):
        lowered = line["text"].lower()
        amounts = AMOUNT_PATTERN.findall(line["text"])
        if not amounts:
            continue
        score = 0.0
        if any(keyword in lowered for keyword in ("grand total", "total", "amount due", "balance due", "cash")):
            score += 3.0
        if "subtotal" in lowered:
            score += 1.0
        score += 0.25 * index
        scored_candidates.append((score, amounts[-1]))
    if not scored_candidates:
        return None
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates[0][1]


def _recover_line_items(token_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    line_items: list[dict[str, Any]] = []
    pending_description: str | None = None
    for line in token_lines:
        text = line["text"].strip()
        if not text or any(keyword in text.lower() for keyword in ("total", "subtotal", "tax", "cash", "change", "date", "invoice")):
            continue
        combined_match = LINE_ITEM_PATTERN.match(text)
        if combined_match:
            line_items.append(
                {
                    "description": _clean_description(combined_match.group("description")),
                    "quantity": _parse_number(combined_match.group("quantity")),
                    "unit_price": _parse_number(combined_match.group("unit_price")),
                }
            )
            pending_description = None
            continue

        if pending_description and re.match(r"^\d+\s*x\s*\d", text, flags=re.IGNORECASE):
            parts = AMOUNT_PATTERN.findall(text)
            quantity_match = re.match(r"^(?P<quantity>\d+(?:\.\d+)?)", text)
            line_items.append(
                {
                    "description": pending_description,
                    "quantity": _parse_number(quantity_match.group("quantity")) if quantity_match else None,
                    "unit_price": _parse_number(parts[-1]) if parts else None,
                }
            )
            pending_description = None
            continue

        amount_matches = AMOUNT_PATTERN.findall(text)
        if amount_matches:
            description = re.sub(AMOUNT_PATTERN, "", text).strip(" x")
            quantity = None
            quantity_prefix = re.match(r"^(?P<quantity>\d+)\s+(?P<desc>.+)$", description)
            if quantity_prefix and not description.lower().startswith(("tel", "fax")):
                quantity = _parse_number(quantity_prefix.group("quantity"))
                description = quantity_prefix.group("desc").strip()
            description = _clean_description(description)
            if description:
                line_items.append(
                    {
                        "description": description,
                        "quantity": quantity,
                        "unit_price": _parse_number(amount_matches[-1]),
                    }
                )
                pending_description = None
                continue

        if _looks_like_description_only(text):
            pending_description = _clean_description(text)
    return line_items


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
        return bool(AMOUNT_PATTERN.search(text))
    if field == "vendor":
        return len(text.split()) >= 2 and not any(character.isdigit() for character in text)
    return True


def _looks_like_description_only(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("subtotal", "total", "tax", "cash", "change", "date", "invoice")):
        return False
    return bool(re.search(r"[A-Za-z]", text)) and not AMOUNT_PATTERN.search(text)


def _clean_description(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -x")
    return text.strip()
