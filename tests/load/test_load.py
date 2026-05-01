from __future__ import annotations

import copy
import logging
import time

import pytest

from document_intelligence_engine.testing import run_concurrent_requests, write_json_report


pytestmark = pytest.mark.asyncio


def _successful_parser(result: dict[str, object], latency_seconds: float):
    def _parse_file(file_path, debug=False):  # noqa: ANN001, FBT002
        _ = file_path
        _ = debug
        time.sleep(latency_seconds)
        return copy.deepcopy(result)

    return _parse_file


async def _upload_document(client, sample_image_bytes: bytes) -> int:
    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    return response.status_code


@pytest.mark.parametrize(
    ("profile_name", "concurrency", "request_count"),
    [
        ("load_10_concurrent_users", 10, 20),
        ("load_50_concurrent_users", 50, 100),
    ],
)
async def test_concurrent_load_profiles(
    api_client,
    sample_image_bytes,
    mock_parser_result,
    monkeypatch,
    tmp_path,
    caplog,
    profile_name,
    concurrency,
    request_count,
):
    client, app = api_client
    caplog.set_level(logging.INFO)
    app.state.runtime.settings.api.rate_limit_per_minute = 0
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _successful_parser(mock_parser_result, latency_seconds=0.01),
    )

    report = await run_concurrent_requests(
        profile_name=profile_name,
        total_requests=request_count,
        concurrency=concurrency,
        request_callable=lambda index: _upload_document(client, sample_image_bytes),
        metadata={"endpoint": "/parse-document"},
    )

    report_path = write_json_report(tmp_path / f"{profile_name}.json", report)

    assert report_path.exists()
    assert report["success_rate"] == 1.0
    assert report["error_rate"] == 0.0
    assert report["status_codes"] == {"200": request_count}
    assert report["response_time_ms"]["p95"] >= report["response_time_ms"]["p50"]
    assert report["response_time_ms"]["max"] >= report["response_time_ms"]["min"]
    assert report["resources"]["enabled"] in {True, False}

    completed_logs = [
        record
        for record in caplog.records
        if record.msg == "request_completed" and getattr(record, "path", None) == "/parse-document"
    ]
    assert len(completed_logs) >= request_count


async def test_burst_traffic_spike(api_client, sample_image_bytes, mock_parser_result, monkeypatch, tmp_path):
    client, app = api_client
    app.state.runtime.settings.api.rate_limit_per_minute = 0
    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        _successful_parser(mock_parser_result, latency_seconds=0.005),
    )

    report = await run_concurrent_requests(
        profile_name="load_burst_spike",
        total_requests=120,
        concurrency=80,
        request_callable=lambda index: _upload_document(client, sample_image_bytes),
        metadata={"endpoint": "/parse-document", "scenario": "burst"},
    )

    report_path = write_json_report(tmp_path / "load_burst_spike.json", report)

    assert report_path.exists()
    assert report["success_rate"] == 1.0
    assert report["error_rate"] == 0.0
    assert report["throughput_rps"] > 0
    assert report["response_time_ms"]["average"] > 0
    assert report["status_codes"]["200"] == 120
