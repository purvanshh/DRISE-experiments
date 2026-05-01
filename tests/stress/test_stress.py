from __future__ import annotations

import copy
import time

import pytest

from api.schemas import ParseDocumentResponse
from document_intelligence_engine.testing import run_concurrent_requests, write_json_report


pytestmark = pytest.mark.asyncio


def _successful_parser(result: dict[str, object], latency_seconds: float, page_count: int | None = None):
    def _parse_file(file_path, debug=False):  # noqa: ANN001, FBT002
        _ = file_path
        _ = debug
        time.sleep(latency_seconds)
        payload = copy.deepcopy(result)
        if page_count is not None:
            payload["metadata"]["page_count"] = page_count
        return payload

    return _parse_file


async def _upload(client, file_name: str, file_bytes: bytes, content_type: str) -> int:
    response = await client.post(
        "/parse-document",
        files={"file": (file_name, file_bytes, content_type)},
    )
    return response.status_code


async def test_very_large_pdf_files_rejected_without_crash(api_client, sample_pdf_bytes, tmp_path):
    client, app = api_client
    app.state.runtime.settings.api.max_upload_size_mb = 1
    large_pdf_bytes = sample_pdf_bytes + (b"A" * (2 * 1024 * 1024))

    report = await run_concurrent_requests(
        profile_name="stress_large_pdf_payloads",
        total_requests=5,
        concurrency=5,
        request_callable=lambda index: _upload(client, "very-large.pdf", large_pdf_bytes, "application/pdf"),
        metadata={"scenario": "large_pdf"},
    )

    report_path = write_json_report(tmp_path / "stress_large_pdf_payloads.json", report)

    assert report_path.exists()
    assert report["status_codes"] == {"413": 5}
    assert report["success_rate"] == 0.0
    assert report["error_rate"] == 1.0
    if report["resources"]["enabled"]:
        assert report["resources"]["memory"]["delta_rss_bytes"] < 64 * 1024 * 1024


async def test_multi_page_documents_20_plus_pages(api_client, sample_pdf_bytes, mock_parser_result, monkeypatch):
    client, app = api_client
    app.state.runtime.settings.api.rate_limit_per_minute = 0
    monkeypatch.setattr("ingestion.file_validator.pdfinfo_from_path", lambda path: {"Pages": 25})
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _successful_parser(mock_parser_result, latency_seconds=0.02, page_count=25),
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("multi-page.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = ParseDocumentResponse.model_validate(response.json())
    assert payload.metadata.page_count == 25
    assert payload.metadata.processing_time_ms >= 0


async def test_repeated_rapid_requests_memory_stable(
    api_client,
    sample_image_bytes,
    mock_parser_result,
    monkeypatch,
    tmp_path,
):
    client, app = api_client
    app.state.runtime.settings.api.rate_limit_per_minute = 0
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _successful_parser(mock_parser_result, latency_seconds=0.01),
    )

    report = await run_concurrent_requests(
        profile_name="stress_repeated_rapid_requests",
        total_requests=75,
        concurrency=25,
        request_callable=lambda index: _upload(client, "sample.png", sample_image_bytes, "image/png"),
        metadata={"scenario": "rapid_requests"},
    )

    report_path = write_json_report(tmp_path / "stress_repeated_rapid_requests.json", report)

    assert report_path.exists()
    assert report["success_rate"] == 1.0
    assert report["error_rate"] == 0.0
    if report["resources"]["enabled"]:
        assert report["resources"]["memory"]["delta_rss_bytes"] < 64 * 1024 * 1024
        assert report["resources"]["memory"]["peak_rss_bytes"] >= report["resources"]["memory"]["initial_rss_bytes"]
