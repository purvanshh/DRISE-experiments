"""OCR service implementations."""

from __future__ import annotations

from typing import cast

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.errors import ConfigurationError, OCRProcessingError
from document_intelligence_engine.domain.contracts import BoundingBox, OCRResult, OCRToken
from document_intelligence_engine.ocr.base import OCRBackend, image_from_bytes


class TesseractOCRBackend:
    backend_name = "tesseract"

    def extract(self, image_bytes: bytes, page_number: int) -> OCRResult:
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRProcessingError("pytesseract is not installed.") from exc

        settings = get_settings()
        if settings.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.ocr.tesseract_cmd
        image = image_from_bytes(image_bytes)
        data = pytesseract.image_to_data(
            image,
            lang=settings.ocr.language,
            output_type=pytesseract.Output.DICT,
        )

        tokens: list[OCRToken] = []
        for index, text in enumerate(data["text"]):
            clean_text = text.strip()
            if not clean_text:
                continue
            confidence = max(float(data["conf"][index]) / 100.0, 0.0)
            if confidence < settings.ocr.min_confidence:
                continue
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            tokens.append(
                OCRToken(
                    text=clean_text,
                    bbox=BoundingBox(x0=left, y0=top, x1=left + width, y1=top + height),
                    confidence=min(confidence, 1.0),
                    page_number=page_number,
                )
            )

        return OCRResult(tokens=tokens, engine=self.backend_name, language=settings.ocr.language)


class OCRService:
    def __init__(self, backend: OCRBackend | None = None) -> None:
        settings = get_settings()
        if backend is not None:
            self._backend = backend
            return
        if settings.ocr.backend == "tesseract":
            self._backend = cast(OCRBackend, TesseractOCRBackend())
            return
        raise ConfigurationError(f"Unsupported OCR backend: {settings.ocr.backend}")

    def extract(self, image_bytes: bytes, page_number: int) -> OCRResult:
        try:
            return self._backend.extract(image_bytes=image_bytes, page_number=page_number)
        except OCRProcessingError:
            raise
        except Exception as exc:
            raise OCRProcessingError("OCR extraction failed.") from exc
