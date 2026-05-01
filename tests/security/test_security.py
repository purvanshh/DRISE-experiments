from __future__ import annotations

import copy
import logging

import pytest

from api.schemas import ErrorResponse, ParseDocumentResponse


pytestmark = pytest.mark.asyncio


def _successful_parser(result: dict[str, object]):
    def _parse_file(file_path, debug=False):  # noqa: ANN001, FBT002
        _ = file_path
        _ = debug
        return copy.deepcopy(result)

    return _parse_file


async def test_invalid_file_type_disguised_as_pdf(api_client, monkeypatch):
    client, _ = api_client

    monkeypatch.setattr(
        "ingestion.file_validator.pdfinfo_from_path",
        lambda path: (_ for _ in ()).throw(RuntimeError("not a pdf")),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("payload.pdf", b"MZ-not-a-pdf", "application/pdf")},
    )

    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "PDF validation failed."


async def test_corrupted_pdf_rejected_safely(api_client, monkeypatch):
    client, _ = api_client

    monkeypatch.setattr(
        "ingestion.file_validator.pdfinfo_from_path",
        lambda path: (_ for _ in ()).throw(RuntimeError("corrupted pdf")),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("corrupted.pdf", b"%PDF-1.7\n%%corrupted", "application/pdf")},
    )

    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "PDF validation failed."


async def test_oversized_payload_rejected_safely(api_client, sample_image_bytes):
    client, app = api_client
    app.state.runtime.settings.api.max_upload_size_mb = 1
    oversized_payload = sample_image_bytes + (b"A" * (2 * 1024 * 1024))

    response = await client.post(
        "/parse-document",
        files={"file": ("oversized.png", oversized_payload, "image/png")},
    )

    assert response.status_code == 413
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Uploaded file exceeds configured size limit."


async def test_path_traversal_attempt_sanitized(api_client, sample_image_bytes, mock_parser_result, monkeypatch):
    client, app = api_client
    monkeypatch.setattr(app.state.runtime.parser_service, "parse_file", _successful_parser(mock_parser_result))

    response = await client.post(
        "/parse-document",
        files={"file": ("../../etc/passwd.png", sample_image_bytes, "image/png")},
    )

    assert response.status_code == 200
    payload = ParseDocumentResponse.model_validate(response.json())
    assert payload.metadata.filename == "passwd.png"
    assert ".." not in payload.metadata.filename
    assert "/" not in payload.metadata.filename


async def test_injection_attempt_in_metadata_sanitized(api_client, sample_image_bytes, mock_parser_result, monkeypatch):
    client, app = api_client
    monkeypatch.setattr(app.state.runtime.parser_service, "parse_file", _successful_parser(mock_parser_result))
    malicious_name = "invoice\r\nX-Injected: evil<script>alert(1)</script>.png"

    response = await client.post(
        "/parse-document",
        files={"file": (malicious_name, sample_image_bytes, "image/png")},
    )

    assert response.status_code == 200
    payload = ParseDocumentResponse.model_validate(response.json())
    assert "\r" not in payload.metadata.filename
    assert "\n" not in payload.metadata.filename
    assert "<" not in payload.metadata.filename
    assert ">" not in payload.metadata.filename
    assert payload.metadata.filename.endswith(".png")


async def test_rate_limiting_throttles_excess_requests(api_client, sample_image_bytes, mock_parser_result, monkeypatch):
    client, app = api_client
    app.state.runtime.settings.api.rate_limit_per_minute = 2
    monkeypatch.setattr(app.state.runtime.parser_service, "parse_file", _successful_parser(mock_parser_result))

    first = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    second = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    third = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    payload = ErrorResponse.model_validate(third.json())
    assert payload.error == "Rate limit exceeded"


async def test_suspicious_inputs_logged(api_client, caplog):
    client, _ = api_client
    caplog.set_level(logging.INFO)

    response = await client.post(
        "/parse-document",
        files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert any(record.msg == "request_received" for record in caplog.records)
    assert any(
        record.msg == "request_completed"
        and getattr(record, "status_code", None) == 400
        and getattr(record, "path", None) == "/parse-document"
        for record in caplog.records
    )
