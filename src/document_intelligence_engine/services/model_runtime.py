"""Model runtime and artifact management."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from document_intelligence_engine.core.config import AppSettings
from document_intelligence_engine.core.logging import get_logger


logger = get_logger(__name__)


class ModelRuntimeError(Exception):
    """Raised when model runtime initialization or inference fails."""


class LayoutAwareModelService:
    """Startup-loaded model service with real LayoutLMv3 inference."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._loaded = False
        self._device = self._resolve_device(settings)
        self._name = settings.model.layoutlmv3_model_name
        self._version = settings.model.version
        self._using_heuristic = False
        self._inference_service = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def device(self) -> str:
        return self._device

    @property
    def using_heuristic(self) -> bool:
        return self._using_heuristic

    def load(self) -> None:
        """Load the model.

        Tries a local fine-tuned checkpoint first. If it is incomplete or
        absent, loads the configured published LayoutLMv3 checkpoint/model.
        """
        checkpoint_path = self._settings.model.checkpoint_path
        use_fallback = getattr(self._settings.model, "use_heuristic_fallback", True)

        resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)

        if resolved_checkpoint is not None:
            self._load_real_model(str(resolved_checkpoint))
        elif checkpoint_path:
            logger.warning(
                "model_checkpoint_incomplete_using_published_model",
                extra={"checkpoint_path": str(checkpoint_path), "model_name": self._name},
            )
            self._load_real_model(None)
        elif use_fallback:
            self._using_heuristic = True
            self._loaded = True
            logger.info(
                "model_loaded_heuristic_fallback",
                extra={"reason": "no_checkpoint", "model_name": self._name},
            )
        else:
            self._load_real_model(None)

        logger.info(
            "model_loaded",
            extra={
                "model_name": self._name,
                "model_version": self._version,
                "device": self._device,
                "mode": "heuristic" if self._using_heuristic else "layoutlmv3",
            },
        )

    def _resolve_checkpoint_path(self, checkpoint_path: str | None) -> Path | None:
        if not checkpoint_path:
            return None

        base_path = Path(checkpoint_path).expanduser().resolve()
        candidates = [
            base_path / "best",
            base_path / "final",
            base_path,
        ]
        for candidate in candidates:
            if self._is_complete_checkpoint(candidate):
                return candidate
        return None

    @staticmethod
    def _is_complete_checkpoint(path: Path) -> bool:
        if not path.exists() or not (path / "config.json").exists():
            return False
        weight_files = (
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "tf_model.h5",
        )
        return any(weight_file.exists() for weight_file in weight_files)

    def _load_real_model(self, checkpoint: str | None) -> None:
        """Initialize LayoutLMv3InferenceService."""
        from document_intelligence_engine.multimodal.layoutlmv3 import (
            LayoutLMv3InferenceService,
        )

        try:
            self._inference_service = LayoutLMv3InferenceService(
                checkpoint_path=checkpoint,
            )
            self._inference_service.load()
            self._using_heuristic = False
            self._loaded = True
        except Exception as exc:
            use_fallback = getattr(self._settings.model, "use_heuristic_fallback", True)
            if use_fallback:
                logger.warning(
                    "model_load_failed_using_heuristic",
                    extra={"error": str(exc)},
                )
                self._using_heuristic = True
                self._loaded = True
            else:
                raise ModelRuntimeError(f"Failed to load model: {exc}") from exc

    def predict(
        self,
        ocr_tokens: list[dict[str, Any]],
        page_image: Image.Image | bytes | None = None,
    ) -> list[dict[str, Any]]:
        if not self._loaded:
            raise ModelRuntimeError("Model runtime is not loaded.")

        if self._using_heuristic:
            return heuristic_predict(ocr_tokens, self._settings.postprocessing.field_aliases)

        # Real model inference
        return self._predict_with_model(ocr_tokens, page_image=page_image, use_layout=True)

    def _predict_with_model(
        self,
        ocr_tokens: list[dict[str, Any]],
        page_image: Image.Image | bytes | None = None,
        *,
        use_layout: bool = True,
    ) -> list[dict[str, Any]]:
        """Run real LayoutLMv3 inference on OCR tokens."""
        from document_intelligence_engine.domain.contracts import (
            BoundingBox,
            OCRResult,
            OCRToken,
        )

        if not ocr_tokens:
            return []

        # Convert dict tokens to OCRResult
        tokens = []
        for token_dict in ocr_tokens:
            bbox_data = token_dict.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_data, list) and len(bbox_data) == 4:
                bbox = BoundingBox(x0=bbox_data[0], y0=bbox_data[1], x1=bbox_data[2], y1=bbox_data[3])
            elif isinstance(bbox_data, dict):
                bbox = BoundingBox(**bbox_data)
            else:
                bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0)

            tokens.append(
                OCRToken(
                    text=str(token_dict.get("text", "")),
                    bbox=bbox,
                    confidence=float(token_dict.get("confidence", 0.0)),
                    page_number=int(token_dict.get("page_number", 1)),
                )
            )

        ocr_result = OCRResult(tokens=tokens, engine="pipeline", language="en")
        prediction = self._inference_service.predict(ocr_result, page_image=page_image, use_layout=use_layout)

        # Convert ModelPrediction to list of dicts for the postprocessing pipeline
        results = []
        for idx, token_dict in enumerate(ocr_tokens):
            label = prediction.labels[idx] if idx < len(prediction.labels) else "O"
            confidence = prediction.confidences[idx] if idx < len(prediction.confidences) else 0.0
            results.append({
                "text": str(token_dict.get("text", "")),
                "label": label,
                "confidence": round(float(confidence), 6),
            })
        return results

    def predict_text_only(self, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict without layout features (for ablation)."""
        if self._using_heuristic:
            return heuristic_predict(ocr_tokens, self._settings.postprocessing.field_aliases)
        return self._predict_with_model(ocr_tokens, use_layout=False)

    def predict_without_postprocessing(self, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.predict(ocr_tokens)

    @staticmethod
    def _resolve_device(settings: AppSettings) -> str:
        requested_device = settings.model.device.lower()
        if requested_device not in {"cpu", "cuda"}:
            return "cpu"
        if requested_device == "cuda":
            try:
                import torch
            except ImportError:
                if settings.model.cpu_fallback:
                    logger.warning("torch_missing_using_cpu_fallback")
                    return "cpu"
                raise ModelRuntimeError("Torch is not installed for CUDA execution.")
            if torch.cuda.is_available():
                return "cuda"
            if settings.model.cpu_fallback:
                logger.warning("cuda_unavailable_using_cpu_fallback")
                return "cpu"
            raise ModelRuntimeError("CUDA requested but no GPU is available.")
        return "cpu"


# ---------------------------------------------------------------------------
# Heuristic fallback (kept for backward compatibility + ablation)
# ---------------------------------------------------------------------------

def heuristic_predict(
    ocr_tokens: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> list[dict[str, Any]]:
    if not ocr_tokens:
        return []

    normalized_tokens = [_normalize_token(token.get("text", "")) for token in ocr_tokens]
    alias_sequences = {
        alias: [part for part in alias.split(" ") if part]
        for alias in sorted(field_aliases, key=lambda item: len(item.split()), reverse=True)
    }
    labels = ["O"] * len(ocr_tokens)
    spans: list[tuple[int, int]] = []

    index = 0
    while index < len(ocr_tokens):
        matched = False
        for alias, alias_parts in alias_sequences.items():
            end_index = index + len(alias_parts)
            if end_index > len(ocr_tokens):
                continue
            if normalized_tokens[index:end_index] == alias_parts:
                spans.append((index, end_index))
                labels[index] = "B-KEY"
                for offset in range(index + 1, end_index):
                    labels[offset] = "I-KEY"
                index = end_index
                matched = True
                break
        if not matched:
            index += 1

    for span_index, (_, span_end) in enumerate(spans):
        next_key_start = spans[span_index + 1][0] if span_index + 1 < len(spans) else len(ocr_tokens)
        value_start = span_end
        while value_start < next_key_start and _normalize_token(ocr_tokens[value_start].get("text", "")) in {"", ":"}:
            value_start += 1
        if value_start >= next_key_start:
            continue
        labels[value_start] = "B-VALUE"
        for value_index in range(value_start + 1, next_key_start):
            if _normalize_token(ocr_tokens[value_index].get("text", "")) == "":
                continue
            labels[value_index] = "I-VALUE"

    return [
        {
            "text": str(token.get("text", "")),
            "label": labels[index],
            "confidence": round(float(token.get("confidence", 0.0)), 6),
        }
        for index, token in enumerate(ocr_tokens)
    ]


def _normalize_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9#]+", "", str(text).lower().strip())
