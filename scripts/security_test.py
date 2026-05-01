from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import httpx

from document_intelligence_engine.testing import ResourceMonitor, write_json_report


Validator = Callable[[int, dict[str, Any]], tuple[bool, list[str]]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run security validation against the document intelligence API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Target API base URL.")
    parser.add_argument("--endpoint", default="/parse-document", help="Document parsing endpoint.")
    parser.add_argument(
        "--sample-image",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "data" / "sample.png"),
        help="Valid image payload used for traversal and metadata injection tests.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--monitor-pid", type=int, help="Optional server process PID for CPU and RSS monitoring.")
    parser.add_argument("--oversized-bytes", type=int, default=30 * 1024 * 1024, help="Oversized payload byte count.")
    parser.add_argument(
        "--rate-limit-attempts",
        type=int,
        default=125,
        help="Number of rapid invalid requests used to probe the rate limiter.",
    )
    parser.add_argument("--header", action="append", default=[], help="Additional HTTP header in KEY:VALUE format.")
    parser.add_argument("--output", help="Optional JSON report output path.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    sample_image_path = Path(args.sample_image).expanduser().resolve()
    valid_image_bytes = sample_image_path.read_bytes()
    oversized_payload = (valid_image_bytes * ((args.oversized_bytes // len(valid_image_bytes)) + 1))[: args.oversized_bytes]
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    headers = _parse_headers(args.header)
    cases = _build_cases(valid_image_bytes, oversized_payload)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=args.timeout, headers=headers) as client:
        async with ResourceMonitor(pid=args.monitor_pid) as monitor:
            health = await _healthcheck(client)
            findings = []
            for case in cases:
                response = await client.post(
                    endpoint,
                    files={"file": (case["filename"], case["payload"], case["content_type"])},
                )
                body = _decode_json(response)
                passed, details = case["validator"](response.status_code, body)
                findings.append(
                    {
                        "name": case["name"],
                        "status": "passed" if passed else "failed",
                        "http_status": response.status_code,
                        "details": details,
                    }
                )
            findings.append(await _probe_rate_limit(client, endpoint, args.rate_limit_attempts))
            resources = monitor.summary()

    passed_count = sum(1 for finding in findings if finding["status"] == "passed")
    failed_count = len(findings) - passed_count
    report = {
        "base_url": args.base_url.rstrip("/"),
        "summary": {
            "cases": len(findings),
            "passed": passed_count,
            "failed": failed_count,
            "vulnerabilities_found": failed_count,
        },
        "service_health": health,
        "findings": findings,
        "resources": resources,
    }

    if args.output:
        write_json_report(args.output, report)
    print(json.dumps(report, indent=2))
    return 1 if failed_count else 0


async def _healthcheck(client: httpx.AsyncClient) -> dict[str, object]:
    try:
        response = await client.get("/health")
        return {
            "status_code": response.status_code,
            "body": _decode_json(response),
        }
    except Exception as exc:  # pragma: no cover - depends on external server availability
        return {"status_code": 0, "body": {"error": str(exc)}}


def _build_cases(valid_image_bytes: bytes, oversized_payload: bytes) -> list[dict[str, Any]]:
    return [
        {
            "name": "disguised_executable_pdf",
            "filename": "payload.pdf",
            "content_type": "application/pdf",
            "payload": b"MZ-not-a-real-pdf",
            "validator": _reject_validator({400, 422}),
        },
        {
            "name": "corrupted_pdf",
            "filename": "corrupted.pdf",
            "content_type": "application/pdf",
            "payload": b"%PDF-1.7\n%%corrupted",
            "validator": _reject_validator({400, 422}),
        },
        {
            "name": "oversized_payload",
            "filename": "oversized.png",
            "content_type": "image/png",
            "payload": oversized_payload,
            "validator": _reject_validator({413}),
        },
        {
            "name": "path_traversal_filename",
            "filename": "../../etc/passwd.png",
            "content_type": "image/png",
            "payload": valid_image_bytes,
            "validator": _safe_filename_validator(),
        },
        {
            "name": "metadata_injection_filename",
            "filename": "invoice\r\nX-Injected: evil<script>alert(1)</script>.png",
            "content_type": "image/png",
            "payload": valid_image_bytes,
            "validator": _safe_filename_validator(),
        },
    ]


async def _probe_rate_limit(client: httpx.AsyncClient, endpoint: str, attempts: int) -> dict[str, Any]:
    last_status = 0
    for attempt in range(1, attempts + 1):
        response = await client.post(
            endpoint,
            files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
        )
        last_status = response.status_code
        if response.status_code == 429:
            return {
                "name": "rate_limit_exceeded",
                "status": "passed",
                "http_status": 429,
                "details": [f"rate_limited_after={attempt}", "Throttle activated for excessive request volume."],
            }
    return {
        "name": "rate_limit_exceeded",
        "status": "failed",
        "http_status": last_status,
        "details": ["No throttling observed during rate-limit probe."],
    }


def _reject_validator(expected_statuses: set[int]) -> Validator:
    def _validate(status_code: int, body: dict[str, Any]) -> tuple[bool, list[str]]:
        passed = status_code in expected_statuses and "error" in body
        details = [f"http_status={status_code}"]
        if "error" in body:
            details.append(str(body["error"]))
        return passed, details

    return _validate


def _safe_filename_validator() -> Validator:
    def _validate(status_code: int, body: dict[str, Any]) -> tuple[bool, list[str]]:
        details = [f"http_status={status_code}"]
        if status_code >= 500:
            return False, details + ["Server error triggered by malicious filename."]

        if status_code >= 400:
            return True, details + ["Request rejected safely."]

        filename = str(body.get("metadata", {}).get("filename", ""))
        safe = all(token not in filename for token in ("..", "/", "\\", "\r", "\n", "<", ">"))
        details.append(f"sanitized_filename={filename}")
        return safe, details

    return _validate


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text[:512]}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        key, separator, header_value = value.partition(":")
        if not separator:
            raise ValueError(f"Invalid header format: {value}")
        headers[key.strip()] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
