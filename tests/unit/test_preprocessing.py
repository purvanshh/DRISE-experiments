from __future__ import annotations

import numpy as np
from PIL import Image

from preprocessing.image_preprocessing import ImagePreprocessor


def test_image_resizing(settings, monkeypatch):
    monkeypatch.setattr(settings.preprocessing, "resize_max_width", 60)
    monkeypatch.setattr(settings.preprocessing, "resize_max_height", 40)
    monkeypatch.setattr(settings.preprocessing, "noise_reduction", False)
    monkeypatch.setattr(settings.preprocessing, "normalize_pixels", False)

    preprocessor = ImagePreprocessor()
    image = Image.new("RGB", (300, 150), color=(128, 128, 128))

    processed = preprocessor.preprocess(image)

    assert processed.size == (60, 30)


def test_normalization(settings, monkeypatch):
    monkeypatch.setattr(settings.preprocessing, "grayscale", True)
    monkeypatch.setattr(settings.preprocessing, "noise_reduction", False)
    monkeypatch.setattr(settings.preprocessing, "normalize_pixels", True)

    preprocessor = ImagePreprocessor()
    image = Image.fromarray(
        np.array(
            [
                [[10, 10, 10], [20, 20, 20]],
                [[220, 220, 220], [240, 240, 240]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    processed = preprocessor.preprocess(image)
    pixels = np.array(processed)

    assert processed.mode == "L"
    assert int(pixels.min()) == 0
    assert int(pixels.max()) == 255
