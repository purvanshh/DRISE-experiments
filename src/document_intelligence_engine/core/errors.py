"""Custom exception hierarchy."""

from __future__ import annotations


class DocumentEngineError(Exception):
    """Base application error."""


class InvalidInputError(DocumentEngineError):
    """Raised when an uploaded document is invalid or unsafe."""


class OCRProcessingError(DocumentEngineError):
    """Raised when OCR processing fails."""


class ModelInferenceError(DocumentEngineError):
    """Raised when multimodal model inference fails."""


class ConfigurationError(DocumentEngineError):
    """Raised when configuration is invalid."""
