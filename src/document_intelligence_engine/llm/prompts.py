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
    "You are a deterministic document extraction engine. Return only valid JSON that conforms "
    "to the requested schema. Do not add markdown, commentary, explanations, code fences, or "
    "extra keys. If a field is not supported by the source text, return null for scalar fields "
    "and [] for line_items. Never infer vendor, invoice number, or dates from unrelated content."
)


EXTRACTION_USER_TEMPLATE = """Extract the requested fields from the OCR text and return only JSON.

Rules:
- Preserve the schema keys exactly.
- Use null for missing scalar fields.
- Use [] for missing line_items.
- For line_items, return objects with description, quantity, and unit_price only.
- Do not guess values that are absent or ambiguous.

Schema:
{schema}

OCR Text:
{ocr_text}
"""


FIELD_EXTRACTION_TEMPLATE = """Extract the field "{field_name}" from the context below.
Return only a valid JSON primitive for scalar values or a JSON array for line_items.
If the field is missing, return null.
Do not return an object wrapper unless the field itself is an object.
Do not include explanations, markdown, or surrounding text.

Context:
{context}
"""


STRICT_EXTRACTION_SYSTEM_PROMPT = EXTRACTION_SYSTEM_PROMPT
STRICT_EXTRACTION_USER_TEMPLATE = EXTRACTION_USER_TEMPLATE
STRICT_FIELD_EXTRACTION_TEMPLATE = FIELD_EXTRACTION_TEMPLATE
