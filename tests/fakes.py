from __future__ import annotations

from PIL import Image


class FakeOCREngine:
    def __init__(self, pages: list[list[dict[str, object]]]) -> None:
        self._pages = [list(page) for page in pages]
        self._cursor = 0

    def extract_tokens(self, image: Image.Image) -> list[dict[str, object]]:
        _ = image
        if self._cursor >= len(self._pages):
            return []
        page = self._pages[self._cursor]
        self._cursor += 1
        return [dict(token) for token in page]

    def extract_batch_tokens(self, images: list[Image.Image]) -> list[list[dict[str, object]]]:
        return [self.extract_tokens(image) for image in images]
