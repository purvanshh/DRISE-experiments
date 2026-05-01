from __future__ import annotations

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from tests.assertions import assert_document_schema, assert_parser_result_schema
from tests.fakes import FakeOCREngine


def _build_parser_service() -> DocumentParserService:
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    return DocumentParserService(settings, model_service)


def test_full_pipeline_valid_document(sample_image_path, mock_ocr_output, monkeypatch):
    parser_service = _build_parser_service()

    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: FakeOCREngine([mock_ocr_output]),
    )

    result = parser_service.parse_file(sample_image_path, debug=True)

    assert_parser_result_schema(result)
    assert_document_schema(result["document"], required_fields=("invoice_number", "date", "total_amount"))
    assert result["document"]["invoice_number"]["value"] == "INV-1023"
    assert result["document"]["date"]["value"] == "2025-01-12"
    assert result["document"]["total_amount"]["value"] == 1200.5
    assert result["metadata"]["page_count"] == 1
    assert result["metadata"]["warnings"] == []
    assert result["debug"]["ocr_tokens"]


def test_full_pipeline_noisy_ocr(sample_image_path, noisy_ocr_output, monkeypatch):
    parser_service = _build_parser_service()

    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: FakeOCREngine([noisy_ocr_output]),
    )

    result = parser_service.parse_file(sample_image_path)

    assert_parser_result_schema(result)
    assert "noisy_ocr_output" in result["metadata"]["warnings"]
    assert result["document"]["_errors"]


def test_full_pipeline_missing_fields(sample_image_path, missing_fields_ocr_output, monkeypatch):
    parser_service = _build_parser_service()

    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: FakeOCREngine([missing_fields_ocr_output]),
    )

    result = parser_service.parse_file(sample_image_path)

    assert_parser_result_schema(result)
    assert "missing_required_fields" in result["metadata"]["warnings"]
    assert result["document"]["_errors"]
