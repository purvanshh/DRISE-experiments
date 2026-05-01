from __future__ import annotations

from PIL import Image

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


def test_multi_page_document(conflicting_multi_page_ocr_output, sample_pdf_path, monkeypatch):
    parser_service = _build_parser_service()

    monkeypatch.setattr("ingestion.file_validator.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: [
            Image.new("RGB", (120, 80), color="white"),
            Image.new("RGB", (120, 80), color="white"),
        ],
    )
    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: FakeOCREngine(conflicting_multi_page_ocr_output),
    )

    result = parser_service.parse_file(sample_pdf_path)

    assert_parser_result_schema(result)
    assert_document_schema(result["document"], required_fields=("invoice_number", "date", "total_amount"))
    assert result["metadata"]["page_count"] == 2
    assert "multi_page_inconsistency" in result["metadata"]["warnings"]


def test_model_pipeline_output_schema(sample_image_path, mock_ocr_output, monkeypatch):
    parser_service = _build_parser_service()

    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: FakeOCREngine([mock_ocr_output]),
    )

    result = parser_service.parse_file(sample_image_path, debug=True)

    assert_parser_result_schema(result)
    assert set(result) >= {"document", "metadata", "debug"}
    assert set(result["metadata"]["timing"]) >= {
        "validation",
        "load",
        "preprocessing",
        "ocr",
        "bbox_alignment",
        "model",
        "postprocessing",
        "total",
    }
    assert result["debug"]["page_stats"]
    assert result["debug"]["raw_predictions"]
