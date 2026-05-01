from __future__ import annotations

from typing import Any


TIMING_KEYS = {
    "validation",
    "load",
    "preprocessing",
    "ocr",
    "bbox_alignment",
    "model",
    "postprocessing",
    "total",
}


def assert_document_schema(document: dict[str, Any], required_fields: tuple[str, ...] = ()) -> None:
    assert isinstance(document, dict)
    assert isinstance(document.get("_errors"), list)
    assert isinstance(document.get("_constraint_flags"), list)

    payload_fields = [field_name for field_name in document if not field_name.startswith("_")]
    for field_name in payload_fields:
        assert_field_payload(document[field_name])

    for field_name in required_fields:
        assert field_name in document
        assert_field_payload(document[field_name])


def assert_field_payload(payload: dict[str, Any]) -> None:
    assert isinstance(payload, dict)
    assert "confidence" in payload
    assert "valid" in payload
    assert isinstance(payload["confidence"], (int, float))
    assert isinstance(payload["valid"], bool)


def assert_parser_result_schema(result: dict[str, Any]) -> None:
    assert set(result) >= {"document", "metadata"}
    assert_document_schema(result["document"])
    assert_metadata_schema(result["metadata"])


def assert_metadata_schema(metadata: dict[str, Any]) -> None:
    assert isinstance(metadata["filename"], str)
    assert isinstance(metadata["page_count"], int)
    assert isinstance(metadata["ocr_token_count"], int)
    assert isinstance(metadata["warnings"], list)
    assert set(metadata["timing"]) >= TIMING_KEYS

    confidence_summary = metadata["confidence_summary"]
    assert set(confidence_summary) >= {
        "average",
        "minimum",
        "maximum",
        "kept_fields",
        "dropped_fields",
    }
    assert isinstance(confidence_summary["average"], (int, float))
    assert isinstance(confidence_summary["minimum"], (int, float))
    assert isinstance(confidence_summary["maximum"], (int, float))
    assert isinstance(confidence_summary["kept_fields"], int)
    assert isinstance(confidence_summary["dropped_fields"], int)

    model = metadata["model"]
    assert isinstance(model["name"], str)
    assert isinstance(model["version"], str)
    assert isinstance(model["device"], str)
