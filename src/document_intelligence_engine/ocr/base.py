"""OCR interfaces."""

from __future__ import annotations

from io import BytesIO
from typing import Protocol

from PIL import Image

from document_intelligence_engine.domain.contracts import OCRResult, OCRToken


class OCRBackend(Protocol):
    backend_name: str

    def extract(self, image_bytes: bytes, page_number: int) -> OCRResult:
        """Extract OCR tokens from a single page image."""


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def empty_result(backend_name: str, page_number: int) -> OCRResult:
    _ = page_number
    return OCRResult(tokens=[], engine=backend_name, language="unknown")
