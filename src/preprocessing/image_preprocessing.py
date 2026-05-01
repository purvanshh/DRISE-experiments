"""Configurable image preprocessing for OCR and multimodal models."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from document_intelligence_engine.core.config import get_settings


class ImagePreprocessor:
    """Configurable preprocessing pipeline for page images."""

    def __init__(self) -> None:
        self._settings = get_settings().preprocessing

    def preprocess(self, image: Image.Image) -> Image.Image:
        processed = image.convert("RGB")
        processed = self._resize_preserving_aspect_ratio(processed)

        array = np.array(processed)
        if self._settings.grayscale:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        if self._settings.noise_reduction:
            kernel_size = self._normalized_kernel_size(self._settings.median_blur_kernel_size)
            array = cv2.medianBlur(array, kernel_size)

        if self._settings.deskew:
            array = self._deskew(array)

        if self._settings.normalize_pixels:
            array = cv2.normalize(array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        mode = "L" if array.ndim == 2 else "RGB"
        return Image.fromarray(array.astype(np.uint8), mode=mode)

    def _resize_preserving_aspect_ratio(self, image: Image.Image) -> Image.Image:
        max_width = self._settings.resize_max_width
        max_height = self._settings.resize_max_height
        width, height = image.size
        scale = min(max_width / width, max_height / height, 1.0)
        new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
        if new_size == image.size:
            return image
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _normalized_kernel_size(kernel_size: int) -> int:
        if kernel_size < 1:
            return 1
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def _deskew(self, image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 3:
            grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = image_array

        inverted = cv2.bitwise_not(grayscale)
        coordinates = np.column_stack(np.where(inverted > 0))
        if coordinates.size == 0:
            return image_array

        angle = cv2.minAreaRect(coordinates)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1:
            return image_array

        height, width = grayscale.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_value = 255 if image_array.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(
            image_array,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
