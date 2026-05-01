from __future__ import annotations

from pathlib import Path

import api.main as api_main
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from api.dependencies import RuntimeState
from api.main import create_app
from api.middleware import RateLimitMiddleware
from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from ocr.ocr_engine import OCREngine
from tests.fakes import FakeOCREngine


DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(autouse=True)
def reset_test_state() -> None:
    get_settings.cache_clear()
    OCREngine.reset_instance()
    RateLimitMiddleware._buckets.clear()
    yield
    RateLimitMiddleware._buckets.clear()
    OCREngine.reset_instance()
    get_settings.cache_clear()


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def sample_image_path(data_dir: Path) -> Path:
    return data_dir / "sample.png"


@pytest.fixture
def sample_image_bytes(sample_image_path: Path) -> bytes:
    return sample_image_path.read_bytes()


@pytest.fixture
def sample_pdf_path(data_dir: Path) -> Path:
    return data_dir / "sample.pdf"


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path: Path) -> bytes:
    return sample_pdf_path.read_bytes()


@pytest.fixture
def corrupted_file_path(data_dir: Path) -> Path:
    return data_dir / "corrupted.png"


@pytest.fixture
def mock_ocr_output() -> list[dict[str, object]]:
    return [
        {"text": "Invoice", "bbox": [10, 10, 80, 30], "confidence": 0.99},
        {"text": "Number", "bbox": [90, 10, 170, 30], "confidence": 0.98},
        {"text": "INV-1023", "bbox": [180, 10, 280, 30], "confidence": 0.97},
        {"text": "Date", "bbox": [10, 50, 60, 70], "confidence": 0.96},
        {"text": "2025-01-12", "bbox": [70, 50, 170, 70], "confidence": 0.95},
        {"text": "Total", "bbox": [10, 90, 70, 110], "confidence": 0.94},
        {"text": "Amount", "bbox": [80, 90, 160, 110], "confidence": 0.93},
        {"text": "1200.50", "bbox": [170, 90, 260, 110], "confidence": 0.92},
    ]


@pytest.fixture
def noisy_ocr_output() -> list[dict[str, object]]:
    return [
        {"text": "Inv0ice", "bbox": [10, 10, 50, 28], "confidence": 0.32},
        {"text": "??", "bbox": [60, 10, 85, 28], "confidence": 0.21},
    ]


@pytest.fixture
def missing_fields_ocr_output() -> list[dict[str, object]]:
    return [
        {"text": "Date", "bbox": [10, 10, 60, 28], "confidence": 0.96},
        {"text": "2025-01-12", "bbox": [70, 10, 180, 28], "confidence": 0.95},
    ]


@pytest.fixture
def conflicting_multi_page_ocr_output(mock_ocr_output: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    return [
        mock_ocr_output,
        [
            {"text": "Invoice", "bbox": [10, 10, 80, 30], "confidence": 0.99},
            {"text": "Number", "bbox": [90, 10, 170, 30], "confidence": 0.98},
            {"text": "INV-2048", "bbox": [180, 10, 280, 30], "confidence": 0.97},
        ],
    ]


@pytest.fixture
def mock_model_output() -> list[dict[str, object]]:
    return [
        {"text": "Invoice", "label": "B-KEY", "confidence": 0.99},
        {"text": "Number", "label": "I-KEY", "confidence": 0.98},
        {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.97},
        {"text": "Date", "label": "B-KEY", "confidence": 0.96},
        {"text": "2025-01-12", "label": "B-VALUE", "confidence": 0.95},
        {"text": "Total", "label": "B-KEY", "confidence": 0.94},
        {"text": "Amount", "label": "I-KEY", "confidence": 0.93},
        {"text": "1200.50", "label": "B-VALUE", "confidence": 0.92},
    ]


@pytest.fixture
def mock_structured_output() -> dict[str, object]:
    return {
        "invoice_number": {"value": "INV-1023", "confidence": 0.97, "valid": True},
        "date": {"value": "2025-01-12", "confidence": 0.955, "valid": True},
        "total_amount": {"value": 1200.5, "confidence": 0.93, "valid": True},
        "_errors": [],
        "_constraint_flags": [],
    }


@pytest.fixture
def mock_parser_result(mock_structured_output: dict[str, object]) -> dict[str, object]:
    return {
        "document": mock_structured_output,
        "metadata": {
            "filename": "sample.png",
            "page_count": 1,
            "ocr_token_count": 8,
            "confidence_summary": {
                "average": 0.951667,
                "minimum": 0.93,
                "maximum": 0.97,
                "kept_fields": 3,
                "dropped_fields": 0,
            },
            "timing": {
                "validation": 1.0,
                "load": 1.0,
                "preprocessing": 1.0,
                "ocr": 1.0,
                "bbox_alignment": 1.0,
                "model": 1.0,
                "postprocessing": 1.0,
                "total": 7.0,
            },
            "warnings": [],
            "model": {
                "name": "microsoft/layoutlmv3-base",
                "version": "0.1.0",
                "device": "cpu",
            },
        },
    }


@pytest.fixture
def fake_ocr_engine_factory():
    def _build(pages: list[list[dict[str, object]]]) -> FakeOCREngine:
        return FakeOCREngine(pages)

    return _build


@pytest.fixture
def runtime_state(settings) -> RuntimeState:
    model_service = LayoutAwareModelService(settings)
    parser_service = DocumentParserService(settings, model_service)
    return RuntimeState(
        settings=settings,
        model_service=model_service,
        parser_service=parser_service,
        model_loaded=True,
        ocr_loaded=True,
    )


@pytest_asyncio.fixture
async def api_client(monkeypatch: pytest.MonkeyPatch, runtime_state: RuntimeState):
    monkeypatch.setattr(api_main, "build_runtime", lambda: runtime_state)
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    app = create_app()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app
