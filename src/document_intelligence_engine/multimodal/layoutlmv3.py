"""LayoutLMv3 inference wrapper.

Loads a fine-tuned (or base) LayoutLMv3ForTokenClassification model and runs
real forward-pass inference, producing BIO labels + softmax confidences.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.errors import ModelInferenceError
from document_intelligence_engine.core.logging import get_logger
from document_intelligence_engine.domain.contracts import ModelPrediction, OCRResult
from document_intelligence_engine.multimodal.cord_dataset import ID2LABEL, LABEL2ID, NUM_LABELS

logger = get_logger(__name__)
INVOICE_MODEL_NAME = "jinhybr/OCR-LayoutLMv3-Invoice"


class LayoutLMv3InferenceService:
    """Real LayoutLMv3 inference service.

    Loads a fine-tuned checkpoint (or the base model) and produces
    per-token BIO classification with softmax confidence scores.
    """

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self._settings = get_settings()
        self._device = torch.device(self._settings.model.device)
        self._checkpoint_path = checkpoint_path
        self._model: LayoutLMv3ForTokenClassification | None = None
        self._processor: LayoutLMv3Processor | None = None
        self._model_id2label: dict[int, str] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the published invoice checkpoint and processor."""
        try:
            logger.info("loading_layoutlmv3", extra={"source": INVOICE_MODEL_NAME})

            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                INVOICE_MODEL_NAME,
            )
            self._model.to(self._device)
            self._model.eval()
            self._model_id2label = {
                int(label_id): label for label_id, label in self._model.config.id2label.items()
            }

            self._processor = LayoutLMv3Processor.from_pretrained(
                INVOICE_MODEL_NAME,
                apply_ocr=False,
            )

            self._loaded = True
            logger.info(
                "layoutlmv3_loaded",
                extra={"source": INVOICE_MODEL_NAME, "device": str(self._device)},
            )
        except Exception as exc:
            raise ModelInferenceError(f"Failed to load LayoutLMv3: {exc}") from exc

    @torch.no_grad()
    def predict(
        self,
        ocr_result: OCRResult,
        page_image: Image.Image | bytes | None = None,
    ) -> ModelPrediction:
        """Run inference on OCR tokens + optional page image.

        Args:
            ocr_result: Extracted OCR tokens with bounding boxes.
            page_image: The source page image (PIL Image or raw bytes).
                        If None, a blank 224x224 image is used.

        Returns:
            ModelPrediction with real BIO labels and softmax confidences.
        """
        if not self._loaded or self._model is None or self._processor is None:
            raise ModelInferenceError("LayoutLMv3 model is not loaded.")

        try:
            # Build inputs
            words, boxes = self._extract_words_and_boxes(ocr_result)
            image = self._resolve_image(page_image)

            if not words:
                return ModelPrediction(
                    labels=[], confidences=[], entities={}, model_name=self._settings.model.layoutlmv3_model_name,
                )

            # Normalize bboxes to 0-1000 range
            img_width, img_height = image.size
            normalized_boxes = [self._normalize_bbox(b, img_width, img_height) for b in boxes]

            encoding = self._processor(
                image,
                words,
                boxes=normalized_boxes,
                max_length=self._settings.model.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            encoding = {k: v.to(self._device) for k, v in encoding.items()}

            # Forward pass
            outputs = self._model(**encoding)
            logits = outputs.logits  # (1, seq_len, num_labels)

            # Softmax confidences
            probs = torch.softmax(logits, dim=-1)  # (1, seq_len, num_labels)
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,)
            max_probs = probs.squeeze(0).gather(1, pred_ids.unsqueeze(1)).squeeze(1)  # (seq_len,)

            # Map back to word-level labels
            word_ids = encoding.get("word_ids", None)
            if word_ids is None:
                # Fallback: use the processor's encoding to get word_ids
                word_ids_list = self._processor(
                    image, words, boxes=normalized_boxes,
                    max_length=self._settings.model.max_sequence_length,
                    padding="max_length", truncation=True,
                ).word_ids(batch_index=0)
            else:
                word_ids_list = word_ids

            labels, confidences = self._aggregate_word_predictions(
                pred_ids.cpu(), max_probs.cpu(), word_ids_list, len(words),
            )

            entities = {
                "document_type": "receipt",
                "token_count": len(words),
                "value_tokens": sum(1 for l in labels if "VALUE" in l),
                "key_tokens": sum(1 for l in labels if "KEY" in l),
            }

            return ModelPrediction(
                labels=labels,
                confidences=confidences,
                entities=entities,
                model_name=self._settings.model.layoutlmv3_model_name,
            )

        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"LayoutLMv3 inference failed: {exc}") from exc

    @staticmethod
    def _extract_words_and_boxes(ocr_result: OCRResult) -> tuple[list[str], list[list[int]]]:
        """Extract word texts and bounding boxes from OCRResult."""
        words = []
        boxes = []
        for token in ocr_result.tokens:
            words.append(token.text)
            boxes.append([token.bbox.x0, token.bbox.y0, token.bbox.x1, token.bbox.y1])
        return words, boxes

    @staticmethod
    def _normalize_bbox(box: list[int], width: int, height: int) -> list[int]:
        """Normalize bounding box to 0-1000 range."""
        if width <= 0 or height <= 0:
            return [0, 0, 0, 0]
        return [
            max(0, min(1000, int(1000 * box[0] / width))),
            max(0, min(1000, int(1000 * box[1] / height))),
            max(0, min(1000, int(1000 * box[2] / width))),
            max(0, min(1000, int(1000 * box[3] / height))),
        ]

    @staticmethod
    def _resolve_image(page_image: Image.Image | bytes | None) -> Image.Image:
        """Convert image input to PIL Image."""
        if page_image is None:
            return Image.new("RGB", (224, 224), color="white")
        if isinstance(page_image, bytes):
            return Image.open(BytesIO(page_image)).convert("RGB")
        return page_image.convert("RGB")

    def _aggregate_word_predictions(
        self,
        pred_ids: torch.Tensor,
        max_probs: torch.Tensor,
        word_ids: list[int | None],
        num_words: int,
    ) -> tuple[list[str], list[float]]:
        """Aggregate subword-level predictions back to word level.

        For each word, takes the prediction from its first subword token.
        """
        labels = ["O"] * num_words
        confidences = [0.0] * num_words
        seen_words: set[int] = set()

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            if word_id >= num_words:
                continue
            seen_words.add(word_id)
            raw_label = self._model_id2label.get(pred_ids[token_idx].item(), ID2LABEL.get(0, "O"))
            labels[word_id] = LayoutLMv3InferenceService._remap_label(raw_label)
            confidences[word_id] = round(max_probs[token_idx].item(), 6)

        return labels, confidences

    @staticmethod
    def _remap_label(label: str) -> str:
        """Map third-party receipt labels back into the project's BIO schema."""
        if label == "O":
            return "O"
        if label.startswith("B-"):
            return "B-VALUE"
        if label.startswith("I-"):
            return "I-VALUE"
        normalized = label.strip().lower()
        if normalized in {"ignore", "others"}:
            return "O"
        if normalized.endswith("_key") or normalized.endswith(".key"):
            return "B-KEY"
        if normalized.endswith("_value") or normalized.endswith(".value"):
            return "B-VALUE"
        if len(ID2LABEL) == NUM_LABELS and label in LABEL2ID:
            return label
        return "O"
