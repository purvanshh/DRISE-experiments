"""End-to-end ingestion and OCR pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.logging import get_logger
from ingestion.exceptions import EmptyOCROutputError
from ingestion.file_validator import validate_file
from ingestion.pdf_loader import load_document_images
from ocr.bbox_alignment import align_tokens_with_boxes
from ocr.ocr_engine import get_ocr_engine
from preprocessing.image_preprocessing import ImagePreprocessor


logger = get_logger(__name__)


def process_document(file_path: str) -> list[dict[str, Any]]:
    return process_document_with_metadata(file_path)["ocr_tokens"]


def process_document_with_metadata(file_path: str, debug: bool = False) -> dict[str, Any]:
    settings = get_settings()
    started_at = time.perf_counter()
    resolved_path = Path(file_path).expanduser().resolve()
    logger.info("file_received", extra={"file_path": str(resolved_path), "stage": "validation"})

    validation_started_at = time.perf_counter()
    validated_path = validate_file(resolved_path)
    validation_time_ms = _elapsed_ms(validation_started_at)
    logger.info(
        "file_validated",
        extra={"file_path": str(validated_path), "stage": "validation", "duration_ms": validation_time_ms},
    )

    load_started_at = time.perf_counter()
    images = load_document_images(validated_path)
    load_time_ms = _elapsed_ms(load_started_at)
    logger.info(
        "document_loaded",
        extra={"file_path": str(validated_path), "page_count": len(images), "stage": "load", "duration_ms": load_time_ms},
    )

    preprocessor = ImagePreprocessor()
    ocr_engine = get_ocr_engine()
    structured_output: list[dict[str, Any]] = []
    preprocessing_time_ms = 0.0
    ocr_time_ms = 0.0
    bbox_time_ms = 0.0
    page_stats: list[dict[str, Any]] = []
    debug_pages: list[dict[str, Any]] = []
    batch_size = max(1, settings.performance.page_batch_size)

    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start : batch_start + batch_size]
        processed_batch: list[Any] = []

        preprocessing_started_at = time.perf_counter()
        for image in batch:
            processed_batch.append(preprocessor.preprocess(image))
        preprocessing_time_ms += _elapsed_ms(preprocessing_started_at)

        ocr_started_at = time.perf_counter()
        batch_raw_tokens = ocr_engine.extract_batch_tokens(processed_batch)
        ocr_time_ms += _elapsed_ms(ocr_started_at)

        bbox_started_at = time.perf_counter()
        for offset, (image, processed_image, raw_tokens) in enumerate(
            zip(batch, processed_batch, batch_raw_tokens, strict=False),
            start=batch_start + 1,
        ):
            aligned_tokens = align_tokens_with_boxes(raw_tokens, processed_image.size)
            page_stats.append(
                {
                    "page_number": offset,
                    "raw_token_count": len(raw_tokens),
                    "aligned_token_count": len(aligned_tokens),
                    "image_size": {"width": processed_image.size[0], "height": processed_image.size[1]},
                }
            )
            for token in aligned_tokens:
                structured_output.append(
                    {
                        "text": token["text"],
                        "bbox": token["bbox"],
                        "confidence": token["confidence"],
                    }
                )
            if debug:
                debug_pages.append(
                    {
                        "page_number": offset,
                        "raw_tokens": raw_tokens[: settings.performance.max_debug_tokens],
                        "aligned_tokens": aligned_tokens[: settings.performance.max_debug_tokens],
                    }
                )
        bbox_time_ms += _elapsed_ms(bbox_started_at)

        for image, processed_image in zip(batch, processed_batch, strict=False):
            if settings.performance.memory_cleanup_enabled and processed_image is not image:
                processed_image.close()
            if settings.performance.memory_cleanup_enabled:
                image.close()

    for page_stat in page_stats:
        logger.info(
            "page_processed",
            extra={
                "file_path": str(validated_path),
                "page_number": page_stat["page_number"],
                "token_count": page_stat["aligned_token_count"],
                "stage": "ocr_pipeline",
            },
        )

    if not structured_output:
        logger.error("empty_ocr_output", extra={"file_path": str(validated_path), "stage": "ocr"})
        raise EmptyOCROutputError(f"OCR returned no tokens for document: {validated_path}")

    total_time_ms = round((time.perf_counter() - started_at) * 1000, 3)
    logger.info(
        "ocr_completed",
        extra={
            "file_path": str(validated_path),
            "total_tokens": len(structured_output),
            "stage": "ocr_pipeline",
            "duration_ms": total_time_ms,
        },
    )
    return {
        "file_path": str(validated_path),
        "page_count": len(page_stats),
        "ocr_tokens": structured_output,
        "page_stats": page_stats,
        "timing": {
            "validation": round(validation_time_ms, 3),
            "load": round(load_time_ms, 3),
            "preprocessing": round(preprocessing_time_ms, 3),
            "ocr": round(ocr_time_ms, 3),
            "bbox_alignment": round(bbox_time_ms, 3),
            "total": total_time_ms,
        },
        "debug": {"pages": debug_pages} if debug else {},
    }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
