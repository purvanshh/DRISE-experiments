"""API dependencies and runtime orchestration."""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile

from document_intelligence_engine.core.config import AppSettings, get_settings
from document_intelligence_engine.core.logging import get_logger
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService, ModelRuntimeError
from ingestion.exceptions import InvalidFileError
from ingestion.file_validator import validate_file
from ocr.ocr_engine import get_ocr_engine


logger = get_logger(__name__)


class APIError(Exception):
    """Base API exception."""

    def __init__(self, message: str, status_code: int, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or []


class InvalidUploadError(APIError):
    """Invalid upload request."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


class PayloadTooLargeError(APIError):
    """Request exceeds configured size limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=413)


class ProcessingError(APIError):
    """Pipeline execution failure."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message, status_code=500, details=details)


@dataclass(slots=True)
class StagedUpload:
    path: Path
    filename: str
    content_type: str
    size_bytes: int


@dataclass(slots=True)
class RuntimeState:
    settings: AppSettings
    model_service: "LayoutAwareModelService"
    parser_service: DocumentParserService
    model_loaded: bool
    ocr_loaded: bool
    startup_error: str | None = None


def build_runtime() -> RuntimeState:
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    parser_service = DocumentParserService(settings, model_service)
    model_loaded = False
    ocr_loaded = False
    startup_error: str | None = None

    try:
        model_service.load()
        model_loaded = True
    except ModelRuntimeError as exc:
        startup_error = str(exc)
        logger.error("model_startup_failure", extra={"error": str(exc)})

    try:
        _ = get_ocr_engine()
        ocr_loaded = True
    except Exception as exc:
        message = str(exc)
        startup_error = message if startup_error is None else f"{startup_error}; {message}"
        logger.error("ocr_startup_failure", extra={"error": message})

    return RuntimeState(
        settings=settings,
        model_service=model_service,
        parser_service=parser_service,
        model_loaded=model_loaded,
        ocr_loaded=ocr_loaded,
        startup_error=startup_error,
    )


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


def get_request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid.uuid4().hex))


async def stage_upload(upload_file: UploadFile, settings: AppSettings) -> StagedUpload:
    safe_name = sanitize_filename(upload_file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in settings.ingestion.supported_extensions:
        raise InvalidUploadError(
            "Invalid file format",
            details=[{"field": "file", "issue": "unsupported_extension", "value": suffix}],
        )

    if upload_file.content_type not in settings.security.allowed_content_types:
        raise InvalidUploadError(
            "Unsupported content type",
            details=[{"field": "file", "issue": "unsupported_content_type", "value": upload_file.content_type}],
        )

    payload = await upload_file.read()
    max_size_bytes = settings.api.max_upload_size_mb * 1024 * 1024
    if not payload:
        raise InvalidUploadError(
            "Uploaded file is empty.",
            details=[{"field": "file", "issue": "empty_payload"}],
        )
    if len(payload) > max_size_bytes:
        raise PayloadTooLargeError("Uploaded file exceeds configured size limit.")

    temp_dir = Path(tempfile.mkdtemp(prefix="api-upload-", dir=settings.paths.upload_dir))
    staged_path = temp_dir / safe_name
    staged_path.write_bytes(payload)

    try:
        validate_file(staged_path)
    except InvalidFileError as exc:
        cleanup_staged_upload(staged_path)
        raise InvalidUploadError(
            str(exc),
            details=[{"field": "file", "issue": "validation_failed", "value": str(exc)}],
        ) from exc

    return StagedUpload(
        path=staged_path,
        filename=safe_name,
        content_type=upload_file.content_type or "application/octet-stream",
        size_bytes=len(payload),
    )


async def process_staged_upload(
    staged_upload: StagedUpload,
    runtime: RuntimeState,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    result = await asyncio.to_thread(runtime.parser_service.parse_file, staged_upload.path, debug)
    metadata = dict(result["metadata"])
    metadata["filename"] = staged_upload.filename
    metadata["content_type"] = staged_upload.content_type
    metadata["size_bytes"] = staged_upload.size_bytes
    metadata["processing_time_ms"] = metadata["timing"]["total"]
    return result["document"], metadata, result.get("debug")


async def process_batch_uploads(
    uploads: list[UploadFile],
    runtime: RuntimeState,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]]:
    semaphore = asyncio.Semaphore(runtime.settings.api.batch_concurrency)

    async def _worker(upload: UploadFile) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        async with semaphore:
            staged = await stage_upload(upload, runtime.settings)
            try:
                return await process_staged_upload(staged, runtime)
            finally:
                cleanup_staged_upload(staged.path)

    return await asyncio.gather(*[_worker(upload) for upload in uploads])


def sanitize_filename(filename: str | None) -> str:
    name = Path(filename or "document").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return sanitized[:128] or "document"


def cleanup_staged_upload(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path.parent, ignore_errors=True)
    except OSError:
        logger.warning("staged_upload_cleanup_failed", extra={"path": str(path)})
