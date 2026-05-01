from __future__ import annotations

from postprocessing.entity_grouping import group_entities
from postprocessing.normalization import normalize_entities
from postprocessing.pipeline import postprocess_predictions
from tests.assertions import assert_document_schema


def test_entity_grouping(settings, mock_model_output):
    grouped_entities, errors = group_entities(mock_model_output, settings.postprocessing.field_aliases)

    assert errors == []
    assert grouped_entities == [
        {
            "key": "Invoice Number",
            "field": "invoice_number",
            "value": "INV-1023",
            "confidence": 0.9775,
            "key_confidence": 0.985,
            "value_confidence": 0.97,
        },
        {
            "key": "Date",
            "field": "date",
            "value": "2025-01-12",
            "confidence": 0.955,
            "key_confidence": 0.96,
            "value_confidence": 0.95,
        },
        {
            "key": "Total Amount",
            "field": "total_amount",
            "value": "1200.50",
            "confidence": 0.9275,
            "key_confidence": 0.935,
            "value_confidence": 0.92,
        },
    ]


def test_normalization(settings):
    normalized_entities, errors = normalize_entities(
        [
            {"field": "date", "value": "12/01/2025", "confidence": 0.95, "key": "Date"},
            {"field": "total_amount", "value": "$12O0.50", "confidence": 0.93, "key": "Total Amount"},
        ],
        settings,
    )

    assert errors == []
    assert normalized_entities[0]["value"] == "2025-01-12"
    assert normalized_entities[1]["value"] == 1200.5


def test_postprocessing_output_schema(mock_model_output):
    document = postprocess_predictions(mock_model_output)

    assert_document_schema(document, required_fields=("invoice_number", "date", "total_amount"))
    assert document["invoice_number"]["value"] == "INV-1023"
    assert document["date"]["value"] == "2025-01-12"
    assert document["total_amount"]["value"] == 1200.5
