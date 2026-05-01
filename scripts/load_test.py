from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path

import httpx

from document_intelligence_engine.testing import ResourceMonitor, run_concurrent_requests, write_json_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run concurrent load tests against the document intelligence API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Target API base URL.")
    parser.add_argument("--endpoint", default="/parse-document", help="Document parsing endpoint.")
    parser.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "data" / "sample.png"),
        help="Path to the document file uploaded for each request.",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrent in-flight requests.")
    parser.add_argument("--requests", type=int, default=50, help="Total requests to execute.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--profile", default="load_test", help="Profile name written into the summary report.")
    parser.add_argument("--monitor-pid", type=int, help="Optional server process PID for CPU and RSS monitoring.")
    parser.add_argument("--content-type", help="Override the upload content type.")
    parser.add_argument("--header", action="append", default=[], help="Additional HTTP header in KEY:VALUE format.")
    parser.add_argument("--output", help="Optional JSON report output path.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    file_path = Path(args.file).expanduser().resolve()
    payload = file_path.read_bytes()
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    content_type = args.content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    headers = _parse_headers(args.header)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=args.timeout, headers=headers) as client:
        health = await _healthcheck(client)

        async def _request(request_index: int) -> int:
            _ = request_index
            response = await client.post(
                endpoint,
                files={"file": (file_path.name, payload, content_type)},
            )
            return response.status_code

        report = await run_concurrent_requests(
            profile_name=args.profile,
            total_requests=args.requests,
            concurrency=args.concurrency,
            request_callable=_request,
            timeout_seconds=args.timeout,
            monitor=ResourceMonitor(pid=args.monitor_pid),
            metadata={
                "base_url": args.base_url.rstrip("/"),
                "endpoint": endpoint,
                "file": str(file_path),
                "content_type": content_type,
                "health": health,
            },
        )

    if args.output:
        write_json_report(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


async def _healthcheck(client: httpx.AsyncClient) -> dict[str, object]:
    try:
        response = await client.get("/health")
        return {
            "status_code": response.status_code,
            "body": _decode_json(response),
        }
    except Exception as exc:  # pragma: no cover - depends on external server availability
        return {"status_code": 0, "body": {"error": str(exc)}}


def _decode_json(response: httpx.Response) -> dict[str, object]:
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
