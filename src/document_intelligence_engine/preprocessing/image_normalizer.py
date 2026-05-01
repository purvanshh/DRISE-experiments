"""Preprocessing layer for page images."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.contracts import IngestedPage


class ImageNormalizationService:
    """Deterministic page normalization."""

    def normalize(self, page: IngestedPage) -> IngestedPage:
        settings = get_settings()
        with Image.open(io.BytesIO(page.image_bytes)) as image:
            normalized = image.convert("L" if settings.preprocessing.grayscale else "RGB")
            normalized = ImageOps.exif_transpose(normalized)
            normalized.thumbnail(
                (settings.preprocessing.max_image_side, settings.preprocessing.max_image_side)
            )
            buffer = io.BytesIO()
            normalized.save(buffer, format="PNG")
            return IngestedPage(
                page_number=page.page_number,
                width=normalized.width,
                height=normalized.height,
                image_bytes=buffer.getvalue(),
            )
