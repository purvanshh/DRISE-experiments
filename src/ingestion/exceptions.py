"""Custom exceptions for ingestion and OCR pipeline."""

from __future__ import annotations


class IngestionPipelineError(Exception):
    """Base exception for ingestion pipeline failures."""


class InvalidFileError(IngestionPipelineError):
    """Raised when an input file is invalid or unsupported."""


class PDFLoadingError(IngestionPipelineError):
    """Raised when a PDF cannot be converted into images."""


class OCRExecutionError(IngestionPipelineError):
    """Raised when OCR backend execution fails."""


class EmptyOCROutputError(IngestionPipelineError):
    """Raised when OCR returns no tokens for a processed page or document."""


class BoundingBoxAlignmentError(IngestionPipelineError):
    """Raised when OCR tokens cannot be aligned with valid bounding boxes."""
