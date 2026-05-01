"""PDF and image loading utilities."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from pdf2image import convert_from_path
from PIL import Image

from document_intelligence_engine.core.config import get_settings
from ingestion.exceptions import PDFLoadingError


def load_document_images(file_path: str | Path) -> list[Image.Image]:
    path = Path(file_path).expanduser().resolve()
    extension = path.suffix.lower()

    if extension == ".pdf":
        return _load_pdf_images(path)

    return [_load_image(path)]


def _load_pdf_images(file_path: Path) -> list[Image.Image]:
    settings = get_settings()
    try:
        images = convert_from_path(
            str(file_path),
            dpi=settings.ingestion.pdf_dpi,
            fmt="png",
            thread_count=1,
        )
    except Exception as exc:
        try:
            images = _load_pdf_images_with_fitz(file_path, settings.ingestion.pdf_dpi)
        except Exception as fitz_exc:
            raise PDFLoadingError(f"Unable to convert PDF to images: {file_path}") from fitz_exc

    if not images:
        raise PDFLoadingError(f"PDF produced no pages: {file_path}")

    return [image.convert("RGB") for image in images]


def _load_image(file_path: Path) -> Image.Image:
    with Image.open(file_path) as image:
        return image.convert("RGB")


def _load_pdf_images_with_fitz(file_path: Path, dpi: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    with fitz.open(file_path) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            images.append(image.convert("RGB"))

    return images
