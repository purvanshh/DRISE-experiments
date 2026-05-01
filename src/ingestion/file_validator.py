"""File validation utilities for local document ingestion."""

from __future__ import annotations

from pathlib import Path

import fitz
from pdf2image import pdfinfo_from_path
from PIL import Image, UnidentifiedImageError

from document_intelligence_engine.core.config import get_settings
from ingestion.exceptions import InvalidFileError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_file(file_path: str | Path) -> Path:
    settings = get_settings()
    path = Path(file_path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        raise InvalidFileError(f"Input file does not exist: {path}")

    extension = path.suffix.lower()
    if extension not in set(settings.ingestion.supported_extensions):
        raise InvalidFileError(f"Unsupported file extension: {extension}")

    max_file_size_bytes = settings.ingestion.max_file_size_mb * 1024 * 1024
    if path.stat().st_size == 0 or path.stat().st_size > max_file_size_bytes:
        raise InvalidFileError("Input file exceeds configured size limits or is empty.")

    if extension == ".pdf":
        _validate_pdf(path)
    else:
        _validate_image(path)

    return path


def _validate_pdf(file_path: Path) -> None:
    settings = get_settings()
    try:
        info = pdfinfo_from_path(str(file_path))
        page_count = int(info.get("Pages", 0))
    except Exception as exc:
        try:
            with fitz.open(file_path) as document:
                page_count = document.page_count
        except Exception as fitz_exc:
            raise InvalidFileError("PDF validation failed.") from fitz_exc
    if page_count <= 0:
        raise InvalidFileError("PDF contains no pages.")
    if page_count > settings.security.max_pdf_pages:
        raise InvalidFileError("PDF page count exceeds configured limit.")


def _validate_image(file_path: Path) -> None:
    try:
        with Image.open(file_path) as image:
            image.verify()
        with Image.open(file_path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidFileError("Image validation failed.") from exc

    settings = get_settings()
    if width * height > settings.security.max_image_pixels:
        raise InvalidFileError("Image resolution exceeds configured limit.")
