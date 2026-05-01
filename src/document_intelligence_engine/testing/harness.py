"""Concurrent request harness with optional CPU and memory monitoring."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only when psutil is unavailable
    psutil = None


RequestCallable = Callable[[int], Awaitable[int]]


class ResourceMonitor:
    """Poll CPU and RSS usage while a scenario is running."""

    def __init__(self, pid: int | None = None, interval_seconds: float = 0.05) -> None:
        self._pid = pid or os.getpid()
        self._interval_seconds = interval_seconds
        if psutil is None:
            self._process = None
        else:
            try:
                self._process = psutil.Process(self._pid)
            except psutil.Error:  # pragma: no cover - depends on runtime process availability
                self._process = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._samples: list[dict[str, float | int]] = []

    async def __aenter__(self) -> "ResourceMonitor":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        await self.stop()

    def start(self) -> None:
        if self._process is None or self._running:
            return
        self._running = True
        self._process.cpu_percent(interval=None)
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _sample_loop(self) -> None:
        assert self._process is not None
        started_at = time.perf_counter()
        while self._running:
            try:
                self._samples.append(
                    {
                        "timestamp_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "rss_bytes": int(self._process.memory_info().rss),
                        "cpu_percent": round(float(self._process.cpu_percent(interval=None)), 3),
                    }
                )
            except psutil.Error:  # pragma: no cover - depends on runtime process availability
                self._running = False
                break
            await asyncio.sleep(self._interval_seconds)

    def summary(self) -> dict[str, Any]:
        if self._process is None or not self._samples:
            return {
                "enabled": False,
                "pid": self._pid,
                "sample_count": 0,
                "cpu": {},
                "memory": {},
            }

        rss_values = [int(sample["rss_bytes"]) for sample in self._samples]
        cpu_values = [float(sample["cpu_percent"]) for sample in self._samples]
        return {
            "enabled": True,
            "pid": self._pid,
            "sample_count": len(self._samples),
            "cpu": {
                "average_percent": round(sum(cpu_values) / len(cpu_values), 6),
                "peak_percent": round(max(cpu_values), 6),
            },
            "memory": {
                "initial_rss_bytes": rss_values[0],
                "final_rss_bytes": rss_values[-1],
                "delta_rss_bytes": rss_values[-1] - rss_values[0],
                "peak_rss_bytes": max(rss_values),
            },
        }


async def run_concurrent_requests(
    *,
    profile_name: str,
    total_requests: int,
    concurrency: int,
    request_callable: RequestCallable,
    timeout_seconds: float = 30.0,
    success_predicate: Callable[[int], bool] | None = None,
    monitor: ResourceMonitor | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if total_requests < 1:
        raise ValueError("total_requests must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    predicate = success_predicate or (lambda status_code: 200 <= status_code < 300)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for request_index in range(total_requests):
        queue.put_nowait(request_index)

    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    async def _worker() -> None:
        while not queue.empty():
            try:
                request_index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            request_started_at = time.perf_counter()
            try:
                status_code = await asyncio.wait_for(request_callable(request_index), timeout=timeout_seconds)
                latency_ms = round((time.perf_counter() - request_started_at) * 1000, 3)
                results.append(
                    {
                        "request_index": request_index,
                        "status_code": int(status_code),
                        "latency_ms": latency_ms,
                        "success": predicate(int(status_code)),
                        "error": None,
                    }
                )
            except asyncio.TimeoutError:
                results.append(
                    {
                        "request_index": request_index,
                        "status_code": 0,
                        "latency_ms": round((time.perf_counter() - request_started_at) * 1000, 3),
                        "success": False,
                        "error": "timeout",
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on request callable failures
                results.append(
                    {
                        "request_index": request_index,
                        "status_code": 0,
                        "latency_ms": round((time.perf_counter() - request_started_at) * 1000, 3),
                        "success": False,
                        "error": str(exc),
                    }
                )
            finally:
                queue.task_done()

    if monitor is None:
        monitor = ResourceMonitor()

    async with monitor:
        workers = [asyncio.create_task(_worker()) for _ in range(min(concurrency, total_requests))]
        await queue.join()
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    duration_seconds = max(time.perf_counter() - started_at, 0.000001)
    return _build_summary(
        profile_name=profile_name,
        total_requests=total_requests,
        concurrency=concurrency,
        results=results,
        duration_seconds=duration_seconds,
        resources=monitor.summary(),
        metadata=metadata or {},
    )


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_path


def _build_summary(
    *,
    profile_name: str,
    total_requests: int,
    concurrency: int,
    results: list[dict[str, Any]],
    duration_seconds: float,
    resources: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    status_counter = Counter(str(result["status_code"]) for result in results)
    latency_values = [float(result["latency_ms"]) for result in results]
    successes = sum(1 for result in results if result["success"])
    failures = total_requests - successes
    timeouts = sum(1 for result in results if result["error"] == "timeout")
    error_counter = Counter(result["error"] for result in results if result["error"])

    return {
        "profile": profile_name,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "success_rate": round(successes / total_requests, 6),
        "error_rate": round(failures / total_requests, 6),
        "duration_seconds": round(duration_seconds, 6),
        "throughput_rps": round(total_requests / duration_seconds, 6),
        "response_time_ms": {
            "min": round(min(latency_values), 6),
            "average": round(sum(latency_values) / len(latency_values), 6),
            "p50": round(_percentile(latency_values, 50), 6),
            "p95": round(_percentile(latency_values, 95), 6),
            "p99": round(_percentile(latency_values, 99), 6),
            "max": round(max(latency_values), 6),
        },
        "status_codes": dict(status_counter),
        "errors": dict(error_counter),
        "resources": resources,
        "metadata": metadata,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight
