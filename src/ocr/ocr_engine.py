"""PaddleOCR singleton wrapper."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.logging import get_logger
from ingestion.exceptions import OCRExecutionError


logger = get_logger(__name__)


class OCREngine:
    """Singleton OCR engine wrapper."""

    _instance: "OCREngine | None" = None
    _lock = threading.Lock()

    def __init__(self, backend: Any | None = None) -> None:
        self._settings = get_settings().ocr
        self._backend = backend or self._initialize_backend()

    @classmethod
    def get_instance(cls) -> "OCREngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def extract_tokens(self, image: Image.Image) -> list[dict[str, Any]]:
        try:
            image_array = np.array(image.convert("RGB"))
            ocr_result = self._backend.ocr(image_array, cls=self._settings.use_angle_cls)
        except Exception as exc:
            logger.exception("ocr_failure")
            raise OCRExecutionError("OCR engine execution failed.") from exc

        tokens: list[dict[str, Any]] = []
        for page_result in ocr_result or []:
            for line in page_result or []:
                if not line or len(line) != 2:
                    continue
                points, prediction = line
                if not prediction or len(prediction) != 2:
                    continue
                text, confidence = prediction
                if not text or float(confidence) < self._settings.min_confidence:
                    continue
                bbox = _polygon_to_xyxy(points)
                tokens.append(
                    {
                        "text": str(text).strip(),
                        "bbox": bbox,
                        "confidence": round(float(confidence), 6),
                    }
                )

        return tokens

    def extract_batch_tokens(self, images: list[Image.Image]) -> list[list[dict[str, Any]]]:
        return [self.extract_tokens(image) for image in images]

    def _initialize_backend(self) -> Any:
        if self._settings.backend.lower() != "paddleocr":
            raise OCRExecutionError(f"Unsupported OCR backend: {self._settings.backend}")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRExecutionError("PaddleOCR is not installed.") from exc

        logger.info("ocr_engine_initializing", extra={"backend": "paddleocr"})
        return PaddleOCR(
            use_angle_cls=self._settings.use_angle_cls,
            lang=self._settings.language,
            use_gpu=self._settings.use_gpu,
            det_limit_side_len=self._settings.det_limit_side_len,
            show_log=False,
        )


def get_ocr_engine() -> OCREngine:
    settings = get_settings().ocr
    if settings.singleton_enabled:
        return OCREngine.get_instance()
    return OCREngine()


def _polygon_to_xyxy(points: list[list[float]] | tuple[tuple[float, float], ...]) -> list[int]:
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]
