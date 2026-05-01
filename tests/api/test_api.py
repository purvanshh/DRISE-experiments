from __future__ import annotations

import pytest

from api.schemas import ErrorResponse, HealthResponse, ParseDocumentResponse
from tests.assertions import assert_document_schema


pytestmark = pytest.mark.asyncio


async def test_health_endpoint(api_client):
    client, _ = api_client

    response = await client.get("/health")

    assert response.status_code == 200
    payload = HealthResponse.model_validate(response.json())
    assert payload.status == "ok"
    assert payload.model_loaded is True
    assert payload.ocr_loaded is True


async def test_parse_document_valid_file_upload(api_client, mock_parser_result, sample_image_bytes, monkeypatch):
    client, app = api_client

    monkeypatch.setattr(app.state.runtime.parser_service, "parse_file", lambda file_path, debug=False: mock_parser_result)

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 200
    payload = ParseDocumentResponse.model_validate(response.json())
    assert_document_schema(payload.document, required_fields=("invoice_number", "date", "total_amount"))
    assert payload.metadata.filename == "sample.png"
    assert payload.metadata.content_type == "image/png"
    assert payload.metadata.size_bytes == len(sample_image_bytes)


async def test_parse_document_invalid_file_type(api_client):
    client, _ = api_client

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Invalid file format"
    assert payload.details == [{"field": "file", "issue": "unsupported_extension", "value": ".txt"}]


async def test_parse_document_large_file(api_client, sample_image_bytes):
    client, app = api_client
    app.state.runtime.settings.api.max_upload_size_mb = 0

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 413
    payload = ErrorResponse.model_validate(response.json())
    assert payload.code == 413
    assert payload.error == "Uploaded file exceeds configured size limit."


async def test_parse_document_empty_file(api_client):
    client, _ = api_client

    response = await client.post(
        "/parse-document",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Uploaded file is empty."
