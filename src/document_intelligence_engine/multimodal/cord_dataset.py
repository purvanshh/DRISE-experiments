"""Dataset loaders for LayoutLMv3 fine-tuning.

Supports two receipt sources:
  - CORD receipts, mapping their semantic categories to the project's 5-class
    BIO scheme (``O``, ``B-KEY``, ``I-KEY``, ``B-VALUE``, ``I-VALUE``).
  - FUNSD forms, whose QUESTION/ANSWER annotation is the canonical source of
    KEY (question) and VALUE (answer) supervision. Including FUNSD teaches the
    model to detect printed field names (keys) that receipts do not annotate.

Both the ``naver-clova-ix/cord-v2`` format (``ground_truth`` JSON) and the
``katanaml/cord`` format (``words``/``bboxes``/``ner_tags``) are supported.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image, ImageFont
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor

logger = logging.getLogger(__name__)

# Project BIO label scheme consumed by postprocessing.entity_grouping
LABEL_LIST = ["O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# CORD categories that represent field *names* (keys). CORD v2 annotates the
# values associated with each category, and also marks a small number of
# key-like tokens; everything else with a value is a VALUE span.
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

# CORD categories whose words describe line-item *values*. These are used to
# recover per-category semantics even though the BIO label stays B/I-VALUE.
_LINE_ITEM_CATEGORIES = frozenset({
    "menu.nm",
    "menu.unitprice",
    "menu.cnt",
    "menu.discountprice",
    "menu.sub_nm",
    "menu.sub_unitprice",
    "menu.sub_cnt",
})


def _cord_label_to_bio(cord_label: str, is_first_token: bool) -> str:
    """Map a CORD category string to our BIO label.

    CORD labels look like ``menu.nm``, ``total.total_price``, etc. These are
    the *values* printed on the receipt, so every non-O token maps to a VALUE
    span. KEY supervision comes from FUNSD (see ``_parse_funsd_example``).
    """
    if cord_label == "O" or cord_label.startswith("O"):
        return "O"
    return "B-VALUE" if is_first_token else "I-VALUE"


def _parse_cord_example(example: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    """Parse a CORD example into flat token lists.

    Supports both the ``ground_truth`` JSON format (cord-v2) and the
    ``words``/``bboxes``/``ner_tags`` format (katanaml/cord).
    """
    if "ground_truth" in example:
        return _parse_cord_v2_example(example)
    if "ner_tags" in example and "words" in example:
        return _parse_cord_flat_example(example, label_names)
    raise ValueError(
        "Unsupported CORD example: expected 'ground_truth' or 'ner_tags'/'words' columns."
    )


def _parse_cord_v2_example(example: dict[str, Any]) -> dict[str, Any]:
    gt = json.loads(example["ground_truth"])
    gt_parse = gt.get("gt_parse", gt)

    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []

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
                is_first = len(labels) == 0 or labels[-1] == "O"
                labels.append(_cord_label_to_bio(label, is_first))
    else:
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


def _parse_cord_flat_example(example: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    """Parse the katanaml/cord format (``words``, ``bboxes``, ``ner_tags``)."""
    raw_tags = example["ner_tags"]
    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []
    for word, bbox, tag_id in zip(example["words"], example["bboxes"], raw_tags, strict=False):
        text = str(word).strip()
        if not text:
            continue
        if not bbox:
            bbox = [0, 0, 0, 0]
        tag = label_names[tag_id] if isinstance(tag_id, int) and tag_id < len(label_names) else str(tag_id)
        words.append(text)
        boxes.append([int(value) for value in bbox])
        is_first = not labels or labels[-1] == "O"
        labels.append(_cord_label_to_bio(tag, is_first))

    return {"words": words, "boxes": boxes, "bio_labels": labels}


def _parse_funsd_example(example: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    """Parse a FUNSD example into KEY/VALUE BIO labels.

    FUNSD annotates QUESTION spans (field names) and ANSWER spans (values),
    which is exactly the KEY -> VALUE structure the post-processing layer
    expects. QUESTION tokens map to B-KEY/I-KEY, ANSWER tokens to B-VALUE/I-VALUE.
    """
    words: list[str] = []
    boxes: list[list[int]] = []
    labels: list[str] = []
    for word, bbox, tag_id in zip(example["words"], example["bboxes"], example["ner_tags"], strict=False):
        text = str(word).strip()
        if not text:
            continue
        tag = label_names[tag_id] if isinstance(tag_id, int) and tag_id < len(label_names) else str(tag_id)
        prefix, base = _split_bio_prefix(tag)
        if base in ("QUESTION", "ANSWER"):
            bio_base = "KEY" if base == "QUESTION" else "VALUE"
            labels.append(f"{prefix}-{bio_base}")
        else:
            labels.append("O")
        if not bbox:
            bbox = [0, 0, 0, 0]
        words.append(text)
        boxes.append([int(value) for value in bbox])

    return {"words": words, "boxes": boxes, "bio_labels": labels}


def _split_bio_prefix(tag: str) -> tuple[str, str]:
    """Split ``B-QUESTION`` into ``(B, QUESTION)``; ``O`` into ``(O, O)``."""
    if "-" in tag:
        prefix, base = tag.split("-", maxsplit=1)
        if prefix in ("B", "I"):
            return prefix, base
    return "O", tag


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


class _LayoutLMDataset(torch.utils.data.Dataset):
    """Base dataset that tokenizes words/boxes/labels with a LayoutLMv3 processor."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        processor: LayoutLMv3Processor,
        max_length: int = 512,
        label_names: list[str] | None = None,
    ) -> None:
        self._examples = examples
        self._processor = processor
        self._max_length = max_length
        self._label_names = list(label_names or [])

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self._examples[idx]
        image = example.get("image")
        if image is None:
            image_path = str(example.get("image_path", ""))
            if image_path and Path(image_path).expanduser().exists():
                image = Image.open(image_path).convert("RGB")
            else:
                image = _render_image_from_tokens(example)
        else:
            image = image.convert("RGB")
        width, height = image.size

        parsed = self._parse(example)
        words = parsed["words"]
        boxes = parsed["boxes"]
        bio_labels = parsed["bio_labels"]

        if not words:
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

        labels = self._align_labels(encoding, bio_labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": labels,
        }

    def _parse(self, example: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _align_labels(
        self,
        encoding: dict[str, torch.Tensor],
        bio_labels: list[str],
    ) -> torch.Tensor:
        word_ids = encoding.word_ids(batch_index=0)
        aligned = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(bio_labels):
                    aligned.append(LABEL2ID[bio_labels[word_id]])
                else:
                    aligned.append(-100)
            else:
                if word_id < len(bio_labels):
                    label = bio_labels[word_id]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned.append(LABEL2ID.get(label, LABEL2ID["O"]))
                else:
                    aligned.append(-100)
            previous_word_id = word_id

        return torch.tensor(aligned, dtype=torch.long)


class CORDDataset(_LayoutLMDataset):
    def _parse(self, example: dict[str, Any]) -> dict[str, Any]:
        return _parse_cord_example(example, self._label_names)


class FUNSDDataset(_LayoutLMDataset):
    def _parse(self, example: dict[str, Any]) -> dict[str, Any]:
        return _parse_funsd_example(example, self._label_names)


def _feature_label_names(dataset: Dataset) -> list[str]:
    feature = dataset.features.get("ner_tags")
    if feature is None:
        return []
    while hasattr(feature, "feature"):
        feature = feature.feature
    if hasattr(feature, "names"):
        return list(feature.names)
    return []


def _render_image_from_tokens(example: dict[str, Any]) -> Image.Image:
    """Render a white-canvas image from words + bounding boxes.

    Used when a dataset stores ``image_path`` references that are not
    available locally (e.g. the katanaml/cord cache on another machine).
    """
    from PIL import ImageDraw

    canvas_size = 1200
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    words = example.get("words", [])
    bboxes = example.get("bboxes", [])
    for word, bbox in zip(words, bboxes, strict=False):
        text = str(word).strip()
        if not text or not bbox:
            continue
        x0, y0, x1, y1 = [max(0, min(canvas_size - 1, int(value * 1.1))) for value in bbox]
        if x1 <= x0:
            x1 = min(canvas_size - 1, x0 + 40)
        if y1 <= y0:
            y1 = min(canvas_size - 1, y0 + 20)
        draw.rectangle([x0, y0, x1, y1], outline="lightgray", width=1)
        draw.text((x0 + 2, y0 + 1), text, fill="black", font=font)
    return image


def load_cord_dataset(max_train_samples: int | None = None) -> DatasetDict:
    """Load a CORD dataset, preferring cord-v2 and falling back to katanaml/cord."""
    dataset: DatasetDict | None = None
    try:
        dataset = load_dataset("naver-clova-ix/cord-v2")
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to load naver-clova-ix/cord-v2 (%s); trying katanaml/cord", exc)
        dataset = load_dataset("katanaml/cord")

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
    include_funsd: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation DataLoaders for CORD (+ optional FUNSD).

    Returns:
        (train_loader, val_loader, label_list)
    """
    processor = LayoutLMv3Processor.from_pretrained(
        model_name,
        apply_ocr=False,  # We supply our own OCR tokens
    )

    raw_dataset = load_cord_dataset(max_train_samples=max_train_samples)

    cord_train_labels = _feature_label_names(raw_dataset["train"])
    train_dataset = CORDDataset(
        [dict(record) for record in raw_dataset["train"]],
        processor,
        max_length=max_length,
        label_names=cord_train_labels,
    )
    val_dataset = CORDDataset(
        [dict(record) for record in raw_dataset["validation"]],
        processor,
        max_length=max_length,
        label_names=_feature_label_names(raw_dataset["validation"]) or cord_train_labels,
    )

    if include_funsd:
        try:
            funsd = load_dataset("nielsr/funsd")
            funsd_train_labels = _feature_label_names(funsd["train"])
            funsd_train = FUNSDDataset(
                [dict(record) for record in funsd["train"]],
                processor,
                max_length=max_length,
                label_names=funsd_train_labels,
            )
            funsd_val = FUNSDDataset(
                [dict(record) for record in funsd["validation"]],
                processor,
                max_length=max_length,
                label_names=funsd_train_labels,
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, funsd_train])
            val_dataset = torch.utils.data.ConcatDataset([val_dataset, funsd_val])
            logger.info("Included FUNSD: train += %d, val += %d", len(funsd["train"]), len(funsd["validation"]))
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("FUNSD unavailable (%s); continuing with CORD only", exc)

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
