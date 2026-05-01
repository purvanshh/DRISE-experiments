from __future__ import annotations

import logging

import api.main as api_main
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from api.dependencies import RuntimeState
from api.main import create_app
from api.schemas import ErrorResponse, HealthResponse
from document_intelligence_engine.services.model_runtime import ModelRuntimeError
from ingestion.exceptions import OCRExecutionError


pytestmark = pytest.mark.asyncio


def _exception_parser(exc: Exception):
    def _parse_file(file_path, debug=False):  # noqa: ANN001, FBT002
        _ = file_path
        _ = debug
        raise exc

    return _parse_file


async def test_ocr_failure_returns_structured_error(api_client, sample_image_bytes, monkeypatch):
    client, app = api_client
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _exception_parser(OCRExecutionError("OCR engine execution failed.")),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 502
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "OCR engine execution failed."


async def test_model_inference_failure_returns_structured_error(api_client, sample_image_bytes, monkeypatch):
    client, app = api_client
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _exception_parser(ModelRuntimeError("Model runtime unavailable.")),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 503
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Model runtime unavailable."


async def test_timeout_failure_returns_structured_error(api_client, sample_image_bytes, monkeypatch, caplog):
    client, app = api_client
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _exception_parser(TimeoutError("Document processing timed out.")),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 500
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Internal processing error."
    assert any(record.msg == "unhandled_api_exception" for record in caplog.records)


async def test_missing_model_file_degrades_health(monkeypatch, runtime_state: RuntimeState):
    degraded_runtime = RuntimeState(
        settings=runtime_state.settings,
        model_service=runtime_state.model_service,
        parser_service=runtime_state.parser_service,
        model_loaded=False,
        ocr_loaded=True,
        startup_error="Model checkpoint not found: /models/layoutlmv3.bin",
    )

    monkeypatch.setattr(api_main, "build_runtime", lambda: degraded_runtime)
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    app = create_app()

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    payload = HealthResponse.model_validate(response.json())
    assert payload.status == "degraded"
    assert payload.model_loaded is False
    assert payload.ocr_loaded is True
