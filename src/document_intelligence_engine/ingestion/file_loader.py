"""Document loading utilities."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.contracts import IngestedPage, StoredDocument, ValidatedFile


def persist_validated_file(document: ValidatedFile) -> StoredDocument:
    upload_dir = get_settings().paths.upload_dir
    target_path = upload_dir / f"{document.sha256[:12]}_{document.safe_name}"
    target_path.write_bytes(document.payload)
    return StoredDocument(path=target_path, metadata=document)


def load_pages(document: ValidatedFile) -> list[IngestedPage]:
    if document.extension == ".pdf":
        return _load_pdf_pages(document.payload)
    return [_load_image_page(document.payload)]


def _load_pdf_pages(payload: bytes) -> list[IngestedPage]:
    settings = get_settings()
    pages: list[IngestedPage] = []
    with fitz.open(stream=payload, filetype="pdf") as pdf_document:
        matrix = fitz.Matrix(settings.preprocessing.target_dpi / 72.0, settings.preprocessing.target_dpi / 72.0)
        for index, page in enumerate(pdf_document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append(
                IngestedPage(
                    page_number=index,
                    width=image.width,
                    height=image.height,
                    image_bytes=buffer.getvalue(),
                )
            )
    return pages


def _load_image_page(payload: bytes) -> IngestedPage:
    with Image.open(io.BytesIO(payload)) as image:
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="PNG")
        return IngestedPage(
            page_number=1,
            width=rgb_image.width,
            height=rgb_image.height,
            image_bytes=buffer.getvalue(),
        )
