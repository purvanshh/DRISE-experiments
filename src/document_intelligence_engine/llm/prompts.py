"""Prompt templates for experiment pipelines."""

from __future__ import annotations


DEFAULT_EXTRACTION_SCHEMA = """{
  "invoice_number": string | null,
  "date": "YYYY-MM-DD" | null,
  "vendor": string | null,
  "total_amount": number | null,
  "line_items": [{"description": string, "quantity": number, "unit_price": number}]
}"""


EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise document extraction engine. Return only valid JSON that conforms "
    "to the requested schema. Never invent fields that are not supported by the source text."
)


EXTRACTION_USER_TEMPLATE = """Extract the requested fields from the OCR text and return only JSON.

Schema:
{schema}

OCR Text:
{ocr_text}
"""


FIELD_EXTRACTION_TEMPLATE = """Extract the field "{field_name}" from the context below.
Return only a valid JSON primitive for scalar values or a JSON array for line_items.
If the field is missing, return null.

Context:
{context}
"""
