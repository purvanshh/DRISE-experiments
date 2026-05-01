"""CORD dataset loader for LayoutLMv3 fine-tuning.

Downloads the CORD-v2 receipt dataset from HuggingFace, maps its NER labels
to the project's 5-class BIO schema (O, B-KEY, I-KEY, B-VALUE, I-VALUE),
and builds LayoutLMv3Processor-compatible batches.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor

logger = logging.getLogger(__name__)

# Project BIO label scheme consumed by postprocessing.entity_grouping
LABEL_LIST = ["O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# CORD categories that represent field *names* (keys)
# Everything else with a value is treated as a VALUE span
_CORD_KEY_CATEGORIES = frozenset({
    "menu.nm",
    "menu.unitprice",
    "menu.cnt",
    "menu.discountprice",
    "menu.sub_nm",
    "menu.sub_unitprice",
    "menu.sub_cnt",
    "menu.etc",
    "menu.vatyn",
    "sub_total.subtotal_price",
    "sub_total.discount_price",
    "sub_total.service_price",
    "sub_total.othersvc_price",
    "sub_total.tax_price",
    "sub_total.etc",
    "total.total_price",
    "total.total_etc",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.emoneyprice",
    "total.menutype_cnt",
    "total.menuqty_cnt",
})


def _cord_label_to_bio(cord_label: str, is_first_token: bool) -> str:
    """Map a CORD category string to our BIO label.

    CORD labels look like ``menu.nm``, ``total.total_price``, etc.
    We split them into KEY vs VALUE based on whether they describe
    a field-name concept or a field-value concept.

    For simplicity all CORD categories are treated as VALUE (the receipt
    text itself), and we don't attempt to distinguish which tokens are
    field-name headers on the receipt vs their values — CORD annotates
    the *values* associated with each category.  So every non-O token
    maps to B-VALUE / I-VALUE.

    If you later want to also predict KEY spans you'd need to add a
    separate key-detection head or heuristic.
    """
    if cord_label == "O" or cord_label.startswith("O"):
        return "O"
    # All annotated CORD spans are field values
    return "B-VALUE" if is_first_token else "I-VALUE"


def _parse_cord_example(example: dict[str, Any]) -> dict[str, Any]:
    """Parse a single CORD example into flat token lists.

    CORD stores annotations as a JSON string in the ``ground_truth`` column.
    Each word has ``text``, ``quad`` (bounding box corners), and ``label``.
    """
    gt = json.loads(example["ground_truth"])
    gt_parse = gt.get("gt_parse", gt)

    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []

    # CORD v2 format: gt_parse is a list of line groups
    if isinstance(gt_parse, list):
        for line_group in gt_parse:
            if not isinstance(line_group, dict):
                continue
            for word_info in line_group.get("words", []):
                text = word_info.get("text", "").strip()
                if not text:
                    continue
                quad = word_info.get("quad", {})
                x_coords = [int(quad.get(f"x{i}", 0)) for i in range(1, 5)]
                y_coords = [int(quad.get(f"y{i}", 0)) for i in range(1, 5)]
                box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                boxes.append(box)
                words.append(text)
                label = word_info.get("label", "O")
                is_first = len(labels) == 0 or labels[-1] == "O" or label != _prev_raw_label(labels, word_info, line_group)
                labels.append(_cord_label_to_bio(label, is_first))
    else:
        # Fallback for dict-based gt_parse
        for category, entries in gt_parse.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for word_info in entry.get("words", []):
                    text = word_info.get("text", "").strip()
                    if not text:
                        continue
                    quad = word_info.get("quad", {})
                    x_coords = [int(quad.get(f"x{i}", 0)) for i in range(1, 5)]
                    y_coords = [int(quad.get(f"y{i}", 0)) for i in range(1, 5)]
                    box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    boxes.append(box)
                    words.append(text)
                    label = category
                    is_first = not labels or labels[-1] in ("O",)
                    labels.append(_cord_label_to_bio(label, is_first))

    return {"words": words, "boxes": boxes, "bio_labels": labels}


def _prev_raw_label(labels: list[str], word_info: dict, line_group: dict) -> str:
    """Helper to check continuity for BIO tagging."""
    return word_info.get("label", "O")


def _normalize_bbox(box: list[int], width: int, height: int) -> list[int]:
    """Normalize bounding box to 0-1000 range as expected by LayoutLMv3."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


class CORDDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping CORD for LayoutLMv3 token classification."""

    def __init__(
        self,
        hf_dataset: Dataset,
        processor: LayoutLMv3Processor,
        max_length: int = 512,
    ) -> None:
        self._dataset = hf_dataset
        self._processor = processor
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self._dataset[idx]
        image: Image.Image = example["image"].convert("RGB")
        width, height = image.size

        parsed = _parse_cord_example(example)
        words = parsed["words"]
        boxes = parsed["boxes"]
        bio_labels = parsed["bio_labels"]

        if not words:
            # Edge case: empty annotation → produce a dummy sample
            words = ["[EMPTY]"]
            boxes = [[0, 0, 0, 0]]
            bio_labels = ["O"]

        normalized_boxes = [_normalize_bbox(b, width, height) for b in boxes]

        encoding = self._processor(
            image,
            words,
            boxes=normalized_boxes,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        labels = self._align_labels(encoding, bio_labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": labels,
        }

    def _align_labels(
        self,
        encoding: dict[str, torch.Tensor],
        bio_labels: list[str],
    ) -> torch.Tensor:
        """Align word-level BIO labels with subword token positions.

        Tokens that don't correspond to any word (special tokens, padding)
        get label -100 (ignored by CrossEntropyLoss).
        """
        word_ids = encoding.word_ids(batch_index=0)
        aligned = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word → use the word's label
                if word_id < len(bio_labels):
                    aligned.append(LABEL2ID[bio_labels[word_id]])
                else:
                    aligned.append(-100)
            else:
                # Continuation subword → use I- variant or same label
                if word_id < len(bio_labels):
                    label = bio_labels[word_id]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned.append(LABEL2ID.get(label, LABEL2ID["O"]))
                else:
                    aligned.append(-100)
            previous_word_id = word_id

        return torch.tensor(aligned, dtype=torch.long)


def load_cord_dataset(max_train_samples: int | None = None) -> DatasetDict:
    """Download CORD-v2 and optionally limit training samples."""
    dataset = load_dataset("naver-clova-ix/cord-v2")

    if max_train_samples is not None and max_train_samples > 0:
        train_size = min(max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(train_size))
        logger.info("Limited training set to %d samples", train_size)

    logger.info(
        "CORD dataset loaded: train=%d, validation=%d, test=%d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )
    return dataset


def get_cord_dataloaders(
    model_name: str = "microsoft/layoutlmv3-base",
    batch_size: int = 4,
    max_length: int = 512,
    max_train_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation DataLoaders for CORD.

    Returns:
        (train_loader, val_loader, label_list)
    """
    processor = LayoutLMv3Processor.from_pretrained(
        model_name,
        apply_ocr=False,  # We supply our own OCR tokens
    )

    raw_dataset = load_cord_dataset(max_train_samples=max_train_samples)

    train_dataset = CORDDataset(raw_dataset["train"], processor, max_length=max_length)
    val_dataset = CORDDataset(raw_dataset["validation"], processor, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, LABEL_LIST
