"""FastAPI middleware configuration."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging with request ID propagation."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        request_id = request.headers.get(settings.api.request_id_header, f"req-{uuid.uuid4().hex}")
        request.state.request_id = request_id
        started_at = time.perf_counter()

        logger.info(
            "request_received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        response.headers[settings.api.request_id_header] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": duration_ms,
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiting."""

    _lock = threading.Lock()
    _buckets: dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        limit = settings.api.rate_limit_per_minute
        if limit <= 0:
            return await call_next(request)

        client_key = request.client.host if request.client else "unknown"
        now = time.time()
        with self._lock:
            bucket = self._buckets[client_key]
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "code": 429,
                        "request_id": getattr(request.state, "request_id", None),
                        "details": [],
                    },
                )
            bucket.append(now)
        return await call_next(request)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Early request body size rejection based on Content-Length."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        content_length = request.headers.get("content-length")
        if content_length is not None:
            max_size_bytes = settings.api.max_upload_size_mb * 1024 * 1024
            if int(content_length) > max_size_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Uploaded file exceeds configured size limit.",
                        "code": 413,
                        "request_id": getattr(request.state, "request_id", None),
                        "details": [],
                    },
                )
        return await call_next(request)


def configure_middleware(app: FastAPI) -> None:
    settings = get_settings()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
