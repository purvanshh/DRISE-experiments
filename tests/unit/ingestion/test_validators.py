from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image
import pytest

from document_intelligence_engine.core.errors import InvalidInputError
from document_intelligence_engine.ingestion.validators import sanitize_filename, validate_upload


class UploadStub:
    def __init__(self, filename: str, content_type: str, payload: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_upload_accepts_png() -> None:
    upload = UploadStub("invoice.png", "image/png", _png_bytes())
    validated = asyncio.run(validate_upload(upload))  # type: ignore[arg-type]
    assert validated.extension == ".png"
    assert validated.safe_name == "invoice.png"


def test_validate_upload_rejects_invalid_signature() -> None:
    upload = UploadStub("invoice.png", "image/png", b"not-a-real-image")
    with pytest.raises(InvalidInputError):
        asyncio.run(validate_upload(upload))  # type: ignore[arg-type]


def test_sanitize_filename_strips_unsafe_characters() -> None:
    assert sanitize_filename("../../invoice?.pdf") == "invoice_.pdf"
