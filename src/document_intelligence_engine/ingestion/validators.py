"""Input validation and safe file handling."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import fitz
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.errors import InvalidInputError
from document_intelligence_engine.domain.contracts import ValidatedFile


MAGIC_NUMBERS = {
    ".pdf": [b"%PDF-"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
}


def sanitize_filename(filename: str | None) -> str:
    settings = get_settings()
    original = Path(filename or "document").name
    sanitized = re.sub(settings.security.filename_pattern, "_", original).strip("._")
    return sanitized[:128] or "document"


def _detect_extension(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    content_map = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tiff",
    }
    return content_map.get(content_type or "", "")


def _validate_magic(extension: str, payload: bytes) -> None:
    allowed_signatures = MAGIC_NUMBERS.get(extension, [])
    if not allowed_signatures or not any(payload.startswith(signature) for signature in allowed_signatures):
        raise InvalidInputError("Uploaded file signature does not match the declared type.")


def _validate_pdf(payload: bytes) -> None:
    settings = get_settings()
    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            if document.page_count == 0:
                raise InvalidInputError("PDF contains no pages.")
            if document.page_count > settings.security.max_pdf_pages:
                raise InvalidInputError("PDF page count exceeds configured limit.")
    except InvalidInputError:
        raise
    except Exception as exc:
        raise InvalidInputError("Malformed PDF rejected.") from exc


def _validate_image(payload: bytes) -> None:
    settings = get_settings()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if width * height > settings.security.max_image_pixels:
                raise InvalidInputError("Image resolution exceeds configured limit.")
    except InvalidInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInputError("Malformed image rejected.") from exc


async def validate_upload(upload_file: UploadFile) -> ValidatedFile:
    settings = get_settings()
    safe_name = sanitize_filename(upload_file.filename)
    extension = _detect_extension(safe_name, upload_file.content_type)

    if extension not in settings.security.allowed_extensions:
        raise InvalidInputError("Unsupported file extension.")

    if (upload_file.content_type or "") not in settings.security.allowed_content_types:
        raise InvalidInputError("Unsupported content type.")

    payload = await upload_file.read()
    max_size_bytes = settings.api.max_upload_size_mb * 1024 * 1024
    if not payload or len(payload) > max_size_bytes:
        raise InvalidInputError("Uploaded file is empty or exceeds configured size limits.")

    _validate_magic(extension, payload)

    if extension == ".pdf":
        _validate_pdf(payload)
    else:
        _validate_image(payload)

    return ValidatedFile(
        original_name=upload_file.filename or "document",
        safe_name=safe_name,
        content_type=upload_file.content_type or "application/octet-stream",
        extension=extension,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )
