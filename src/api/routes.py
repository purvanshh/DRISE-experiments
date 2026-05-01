"""API routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile

from api.dependencies import (
    InvalidUploadError,
    RuntimeState,
    cleanup_staged_upload,
    get_request_id,
    get_runtime,
    process_batch_uploads,
    process_staged_upload,
    stage_upload,
)
from api.schemas import BatchParseResponse, HealthResponse, ParseDocumentResponse
from document_intelligence_engine import __version__
from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(runtime: RuntimeState = Depends(get_runtime)) -> HealthResponse:
    status = "ok" if runtime.model_loaded and runtime.ocr_loaded else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=runtime.model_loaded,
        ocr_loaded=runtime.ocr_loaded,
        version=__version__,
    )


@router.post("/parse-document", response_model=ParseDocumentResponse, tags=["documents"])
async def parse_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    debug: bool = False,
    runtime: RuntimeState = Depends(get_runtime),
) -> ParseDocumentResponse:
    staged_upload = await stage_upload(file, runtime.settings)
    background_tasks.add_task(cleanup_staged_upload, staged_upload.path)
    document, metadata, _ = await process_staged_upload(staged_upload, runtime, debug=debug)
    metadata["request_id"] = get_request_id(request)
    logger.info(
        "document_parsed",
        extra={"request_id": metadata["request_id"], "filename": staged_upload.filename},
    )
    return ParseDocumentResponse(document=document, metadata=metadata)


@router.post("/parse-batch", response_model=BatchParseResponse, tags=["documents"])
async def parse_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    runtime: RuntimeState = Depends(get_runtime),
) -> BatchParseResponse:
    if not files:
        raise InvalidUploadError("No files were provided.", details=[{"field": "files", "issue": "missing"}])
    if len(files) > runtime.settings.api.max_batch_files:
        raise InvalidUploadError(
            f"Batch size exceeds configured limit of {runtime.settings.api.max_batch_files} files.",
            details=[{"field": "files", "issue": "batch_limit_exceeded", "value": len(files)}],
        )

    results = await process_batch_uploads(files, runtime)
    request_id = get_request_id(request)
    items = []
    for upload_file, (document, metadata, _) in zip(files, results, strict=False):
        metadata["request_id"] = request_id
        items.append(ParseDocumentResponse(document=document, metadata=metadata))
        logger.info(
            "batch_document_parsed",
            extra={"request_id": request_id, "filename": upload_file.filename or metadata["filename"]},
        )
    return BatchParseResponse(items=items)
