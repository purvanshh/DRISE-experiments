"""JSON Schema definitions for experiment outputs."""

from __future__ import annotations


LINE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": ["string", "null"]},
        "quantity": {"type": ["number", "null"]},
        "unit_price": {"type": ["number", "null"]},
    },
    "required": ["description", "quantity", "unit_price"],
    "additionalProperties": True,
}


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"], "format": "date"},
        "vendor": {"type": ["string", "null"]},
        "total_amount": {"type": ["number", "null"]},
        "line_items": {
            "type": "array",
            "items": LINE_ITEM_SCHEMA,
        },
    },
    "required": ["invoice_number", "date", "vendor", "total_amount", "line_items"],
    "additionalProperties": True,
}
