"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import APIError, ProcessingError, build_runtime
from api.middleware import configure_middleware
from api.routes import router
from api.schemas import ErrorResponse
from document_intelligence_engine.core.logging import configure_logging, get_logger
from document_intelligence_engine.services.model_runtime import ModelRuntimeError
from ingestion.exceptions import EmptyOCROutputError, InvalidFileError, OCRExecutionError, PDFLoadingError


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.runtime = build_runtime()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Layout-Aware Document Intelligence Engine API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    configure_middleware(app)
    app.include_router(router)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return _error_response(request, exc.message, exc.status_code, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"field": ".".join(map(str, error["loc"])), "message": error["msg"]} for error in exc.errors()]
        return _error_response(request, "Invalid request payload", 422, details)

    @app.exception_handler(InvalidFileError)
    async def handle_invalid_file(request: Request, exc: InvalidFileError) -> JSONResponse:
        return _error_response(request, str(exc), 400, [])

    @app.exception_handler(PDFLoadingError)
    async def handle_pdf_error(request: Request, exc: PDFLoadingError) -> JSONResponse:
        return _error_response(request, str(exc), 400, [])

    @app.exception_handler(EmptyOCROutputError)
    async def handle_empty_ocr(request: Request, exc: EmptyOCROutputError) -> JSONResponse:
        return _error_response(request, str(exc), 422, [])

    @app.exception_handler(OCRExecutionError)
    async def handle_ocr_error(request: Request, exc: OCRExecutionError) -> JSONResponse:
        return _error_response(request, str(exc), 502, [])

    @app.exception_handler(ModelRuntimeError)
    async def handle_model_runtime_error(request: Request, exc: ModelRuntimeError) -> JSONResponse:
        return _error_response(request, str(exc), 503, [])

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_api_exception", extra={"request_id": getattr(request.state, "request_id", None)})
        processing_error = ProcessingError("Internal processing error.")
        return _error_response(request, processing_error.message, processing_error.status_code, [])

    return app


def _error_response(
    request: Request,
    message: str,
    status_code: int,
    details: list[dict[str, object]],
) -> JSONResponse:
    payload = ErrorResponse(
        error=message,
        code=status_code,
        request_id=getattr(request.state, "request_id", None),
        details=details,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


app = create_app()
