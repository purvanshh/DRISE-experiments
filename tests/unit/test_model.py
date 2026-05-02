from __future__ import annotations

from types import SimpleNamespace

from document_intelligence_engine.domain.contracts import BoundingBox, OCRResult, OCRToken
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService, heuristic_predict


def test_forward_pass_shape_correctness(settings):
    """Verify the model service returns one label/confidence per input token."""
    model_service = LayoutAwareModelService(settings)
    model_service.load()

    predictions = model_service.predict(
        [
            {"text": "Invoice", "bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"text": "INV-1023", "bbox": [10, 0, 20, 10], "confidence": 0.95},
        ]
    )

    assert len(predictions) == 2
    for pred in predictions:
        assert "text" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert pred["label"] in {"O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"}


def test_heuristic_fallback_output_format(settings):
    """Verify heuristic fallback produces expected BIO labels for known aliases."""
    predictions = heuristic_predict(
        [
            {"text": "Invoice", "confidence": 0.99},
            {"text": "Number", "confidence": 0.98},
            {"text": "INV-1023", "confidence": 0.97},
        ],
        field_aliases=settings.postprocessing.field_aliases,
    )

    assert predictions == [
        {"text": "Invoice", "label": "B-KEY", "confidence": 0.99},
        {"text": "Number", "label": "I-KEY", "confidence": 0.98},
        {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.97},
    ]


def test_inference_output_format(settings):
    """Verify model service predict() returns well-formed dicts."""
    model_service = LayoutAwareModelService(settings)
    model_service.load()

    predictions = model_service.predict(
        [
            {"text": "Invoice", "confidence": 0.99},
            {"text": "Number", "confidence": 0.98},
            {"text": "INV-1023", "confidence": 0.97},
        ]
    )

    assert len(predictions) == 3
    for pred in predictions:
        assert set(pred.keys()) >= {"text", "label", "confidence"}
        assert isinstance(pred["confidence"], float)


def test_model_service_using_heuristic_property(settings):
    """Verify the service reports heuristic mode when no checkpoint exists."""
    model_service = LayoutAwareModelService(settings)
    model_service.load()

    # Without a real checkpoint the service should fall back to heuristic
    assert model_service.loaded
    assert model_service.using_heuristic


def test_predict_text_only_uses_real_model_without_layout(settings):
    calls: list[bool] = []

    class StubInferenceService:
        def predict(self, ocr_result, page_image=None, use_layout=True):
            _ = ocr_result, page_image
            calls.append(use_layout)
            return SimpleNamespace(labels=["B-VALUE"], confidences=[0.91])

    model_service = LayoutAwareModelService(settings)
    model_service._loaded = True
    model_service._using_heuristic = False
    model_service._inference_service = StubInferenceService()
    model_service._model_id2label = {0: "O"}

    predictions = model_service.predict_text_only(
        [
            {"text": "INV-1023", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        ]
    )

    assert calls == [False]
    assert predictions == [{"text": "INV-1023", "label": "B-VALUE", "confidence": 0.91}]
