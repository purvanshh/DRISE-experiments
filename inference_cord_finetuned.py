"""Inference module for the locally fine-tuned CORD LayoutLMv3 checkpoint.

Pipeline
--------
    1. Load ``LayoutLMv3ForTokenClassification`` + ``LayoutLMv3Processor``
       (``apply_ocr=False``) from a local checkpoint folder.
    2. Tokenize OCR output (words + 0-1000 boxes) with the processor.
    3. Forward pass -> softmax probabilities per subword token.
    4. Re-align subword predictions back to word level (first subword per
       word) and apply a confidence threshold (low-confidence words -> ``O``).
    5. Group words into KEY -> VALUE field pairs. Reuses the project's
       ``postprocessing.entity_grouping.group_entities`` when importable,
       otherwise falls back to a self-contained grouping implementation.
    6. Parse numeric values with locale-aware separators
       (e.g. Indonesian ``1.591.600,50`` vs US ``1,591,600.50``).
    7. Benchmark on a validation set with seqeval entity-level F1 plus
       token-level precision/recall/F1.

The 5-class BIO label scheme matches the checkpoint's ``config.json``:
    LABEL2ID = {"O": 0, "B-KEY": 1, "I-KEY": 2, "B-VALUE": 3, "I-VALUE": 4}

CLI
---
    python inference_cord_finetuned.py --evaluate \
        --data-path /path/to/validation.jsonl --batch-size 8

    python inference_cord_finetuned.py --tune-threshold \
        --data-path /path/to/validation.jsonl --thresholds 0.5 0.6 0.7 0.8 0.9

    python inference_cord_finetuned.py --image receipt.jpg \
        --words '["Grand", "Total", "1.591.600,50"]' \
        --boxes '[[0,0,100,20],[0,0,100,20],[0,0,100,20]]'

Validation files are JSONL (or JSON) with per-record fields ``image_path``
(or ``image``), ``words``, ``bboxes`` (or ``boxes``) and ``ner_tags``
(or ``labels``/``bio_labels``; integers or BIO strings are both accepted).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the fine-tuned checkpoint folder (config.json, model.safetensors,
# tokenizer files). Override at runtime with the DRISE_MODEL_PATH env var.
MODEL_PATH = str(Path("~/cord_finetuned").expanduser())

# Confidence filter: word predictions with softmax probability below this are
# downgraded to "O". Tune on the validation set (see sweep_confidence_thresholds)
# or update at runtime with ``set_global_threshold``.
CONFIDENCE_THRESHOLD = 0.7

# Active threshold consulted at call time whenever ``confidence_threshold`` is
# not passed explicitly. Initialised to CONFIDENCE_THRESHOLD.
_ACTIVE_THRESHOLD = CONFIDENCE_THRESHOLD


def set_global_threshold(value: float) -> float:
    """Set the module-wide confidence threshold used by ``predict_*`` defaults.

    Any call to ``predict_receipt`` / ``predict_batch`` /
    ``evaluate_validation_set`` that does NOT pass ``confidence_threshold``
    explicitly will use this value.

    Returns the value that was set (for convenience).
    """
    global _ACTIVE_THRESHOLD
    _ACTIVE_THRESHOLD = float(value)
    return _ACTIVE_THRESHOLD


def get_global_threshold() -> float:
    """Return the current module-wide confidence threshold."""
    return _ACTIVE_THRESHOLD


def _resolve_threshold(value: float | None) -> float:
    """Resolve an explicit threshold, defaulting to the active global value."""
    return _ACTIVE_THRESHOLD if value is None else float(value)

# Number parsing locale for extracted values: "id" (dot thousand separator,
# comma decimal), "en" (comma thousand, dot decimal), or "auto" (heuristic).
LOCALE = "id"

MAX_SEQUENCE_LENGTH = 512

# 5-class BIO scheme identical to the checkpoint's label2id/id2label entries.
LABEL2ID = {"O": 0, "B-KEY": 1, "I-KEY": 2, "B-VALUE": 3, "I-VALUE": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: str | Path = MODEL_PATH,
    device: str | None = None,
) -> tuple[LayoutLMv3ForTokenClassification, LayoutLMv3Processor, torch.device]:
    """Load the fine-tuned model + processor from a local checkpoint folder.

    ``apply_ocr=False`` tells the processor not to run its internal OCR, since
    we already supply ``words`` and ``boxes``.
    """
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # LayoutLMv3Processor in transformers < 4.45 requires a
    # preprocessor_config.json; some checkpoints saved with newer transformers
    # only ship the 5.x-style processor_config.json. In that case fall back to
    # the base LayoutLMv3 processor (identical image/token processing; the
    # model weights are unaffected since the processor has no trainable
    # parameters).
    if not (model_path / "preprocessor_config.json").exists():
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
        )
    else:
        processor = LayoutLMv3Processor.from_pretrained(str(model_path), apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_path))
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    model.to(resolved_device)
    return model, processor, resolved_device


# ---------------------------------------------------------------------------
# Box normalisation
# ---------------------------------------------------------------------------


def normalize_bbox(box: list[int], width: int, height: int) -> list[int]:
    """Normalise a pixel-coordinate box to the 0-1000 range LayoutLMv3 expects."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def maybe_normalize_boxes(boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Normalise boxes unless they already use the 0-1000 coordinate system.

    OCR boxes are typically pixel-based (can exceed 1000); normalised boxes
    never do, so ``max(box) <= 1000`` is a safe detector.
    """
    normalized: list[list[int]] = []
    for box in boxes:
        x0, y0, x1, y1 = (int(c) for c in box)
        if max(x0, y0, x1, y1) > 1000:
            normalized.append(normalize_bbox(box, width, height))
        else:
            normalized.append([x0, y0, x1, y1])
    return normalized


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_receipt(
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    *,
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    device: torch.device,
    confidence_threshold: float | None = None,
    max_length: int = MAX_SEQUENCE_LENGTH,
    locale: str = LOCALE,
) -> dict[str, Any]:
    """Run token-classification inference on one receipt.

    Args:
        image: The receipt image (pixel boxes are normalised against its size).
        words: OCR word strings, in reading order.
        boxes: One [x0, y0, x1, y1] box per word (pixels OR 0-1000).
        model, processor, device: Loaded components (from ``load_model``).

    Returns:
        Dict with per-word ``predictions`` (text/label/confidence/category)
        plus the ``key_value_pairs``, ``errors`` and ``nodes`` produced by
        post-processing.
    """
    return predict_batch(
        [image],
        [words],
        [boxes],
        model=model,
        processor=processor,
        device=device,
        confidence_threshold=confidence_threshold,
        max_length=max_length,
        locale=locale,
    )[0]


@torch.no_grad()
def predict_batch(
    images: list[Image.Image],
    all_words: list[list[str]],
    all_boxes: list[list[list[int]]],
    *,
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    device: torch.device,
    confidence_threshold: float | None = None,
    max_length: int = MAX_SEQUENCE_LENGTH,
    locale: str = LOCALE,
) -> list[dict[str, Any]]:
    """Run token-classification inference on a batch of receipts.

    All receipts in one forward pass (padded to ``max_length``). Documents
    without OCR words are skipped by the model and return empty results.
    Returned list is index-aligned with the inputs.
    """
    confidence_threshold = _resolve_threshold(confidence_threshold)
    batch_size = len(images)
    results: list[dict[str, Any]] = [
        {"predictions": [], "key_value_pairs": [], "errors": [], "nodes": None}
        for _ in range(batch_size)
    ]

    # Documents with words only; empty docs stay as empty results.
    valid = [i for i, words in enumerate(all_words) if words]
    if not valid:
        return results

    valid_images = [images[i].convert("RGB") for i in valid]
    all_normalized_boxes = [
        maybe_normalize_boxes(
            all_boxes[i], valid_images[position].width, valid_images[position].height
        )
        for position, i in enumerate(valid)
    ]

    encoding = processor(
        valid_images,
        [list(all_words[i]) for i in valid],
        boxes=all_normalized_boxes,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Capture per-sample alignment before moving tensors off the batch encoding.
    word_ids_by_sample = [encoding.word_ids(batch_index=i) for i in range(len(valid))]
    encoding = {k: v.to(device) for k, v in encoding.items()}

    outputs = model(**encoding)
    probs = F.softmax(outputs.logits, dim=-1)  # [batch, seq_len, num_labels]
    max_probs, pred_ids = probs.max(dim=-1)

    for position, word_ids in enumerate(word_ids_by_sample):
        sample_index = valid[position]
        words = all_words[sample_index]
        num_words = len(words)

        # Map subword predictions back to whole words (first subword per word),
        # matching the alignment used during training (`_align_labels` in
        # multimodal/cord_dataset.py only supervises the first subword).
        predictions: list[dict[str, Any]] = []
        seen_words: set[int] = set()
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            if word_id >= num_words:
                continue
            seen_words.add(word_id)

            label = ID2LABEL.get(int(pred_ids[position, token_idx].item()), "O")
            confidence = float(max_probs[position, token_idx].item())

            # Confidence-based filtering: below the threshold the word
            # contributes nothing (kept as "O" so word alignment with the OCR
            # output holds).
            if confidence < confidence_threshold:
                label = "O"

            predictions.append(
                {
                    "text": str(words[word_id]),
                    "label": label,
                    "confidence": round(confidence, 6),
                    "category": _semantic_category(label),
                }
            )

        key_value_pairs, errors = group_entities(predictions, field_aliases={})

        # Attach the parsed numeric value (locale-aware) to each pair.
        for pair in key_value_pairs:
            if pair.get("value") is not None:
                pair["numeric_value"] = parse_number(str(pair["value"]), locale=locale)

        results[sample_index] = {
            "predictions": predictions,
            "key_value_pairs": key_value_pairs,
            "errors": errors,
            "nodes": None,
        }

    return results


def _semantic_category(label: str) -> str:
    """Expose the entity type as a coarse category for downstream grouping."""
    if "-" in label:
        return label.split("-", maxsplit=1)[1].lower()
    return ""


# ---------------------------------------------------------------------------
# Post-processing: reuse the project's grouping when importable
# ---------------------------------------------------------------------------


def _import_project_grouping() -> Any | None:
    """Try to import ``postprocessing.entity_grouping.group_entities``.

    The project keeps it under ``src/``; add that directory to ``sys.path`` so
    this module works from anywhere in the repo. Returns None if unavailable.
    """
    try:
        from postprocessing.entity_grouping import group_entities  # type: ignore

        return group_entities
    except Exception:
        pass
    src_dir = Path(__file__).resolve().parent / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from postprocessing.entity_grouping import group_entities  # type: ignore

        return group_entities
    except Exception:
        return None


_PROJECT_GROUPING = _import_project_grouping()


def group_entities(
    predictions: list[dict[str, Any]],
    field_aliases: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Group BIO-tagged words into KEY -> VALUE field pairs.

    Primary implementation is the repo's
    ``postprocessing.entity_grouping.group_entities`` (which appends
    missing-value / orphan-value errors); if it cannot be imported, a simple
    self-contained fallback with the same return contract is used.
    """
    aliases = dict(field_aliases or {})
    if _PROJECT_GROUPING is not None:
        return _PROJECT_GROUPING(predictions, aliases)
    return _fallback_group_entities(predictions, aliases)


def _fallback_group_entities(
    predictions: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Minimal KEY -> VALUE grouping used when the repo module is unavailable."""
    grouped: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pending_key: dict[str, Any] | None = None

    for prediction in predictions:
        label = str(prediction.get("label", "O")).upper().strip()
        text = str(prediction.get("text", "")).strip()
        if not text or label in {"O", "I-VALUE"}:
            continue
        if label == "I-KEY" and pending_key is not None:
            pending_key["text"] = f"{pending_key['text']} {text}"
            pending_key["confidences"].append(float(prediction.get("confidence", 0.0)))
            continue
        if label == "B-KEY":
            if pending_key is not None:
                grouped.append(_finalize_key(pending_key, field_aliases))
                errors.append(
                    {"field": "?", "code": "missing_value", "message": "Key has no paired value."}
                )
            pending_key = {
                "text": text,
                "confidence": float(prediction.get("confidence", 0.0)),
                "confidences": [float(prediction.get("confidence", 0.0))],
            }
            continue
        if label in {"B-VALUE", "I-VALUE"}:
            if pending_key is None:
                errors.append(
                    {
                        "field": "_document",
                        "code": "orphan_value",
                        "message": f"Unpaired value '{text}' ignored.",
                    }
                )
                continue
            if label == "I-VALUE" and pending_key.get("value_text"):
                pending_key["value_text"] = f"{pending_key['value_text']} {text}"
                pending_key["value_confidences"].append(float(prediction.get("confidence", 0.0)))
                continue
            pending_key["value_text"] = text
            pending_key["value_confidence"] = float(prediction.get("confidence", 0.0))
            pending_key["value_confidences"] = [float(prediction.get("confidence", 0.0))]

    if pending_key is not None:
        grouped.append(_finalize_key(pending_key, field_aliases))
    return grouped, errors


def _finalize_key(pending_key: dict[str, Any], field_aliases: dict[str, str]) -> dict[str, Any]:
    value = pending_key.get("value_text")
    key_conf = fmean(pending_key["confidences"])
    value_conf = pending_key.get("value_confidence")
    return {
        "key": pending_key["text"],
        "field": _canonicalize_field_name(pending_key["text"], field_aliases),
        "value": value,
        "confidence": (
            round(fmean([key_conf, value_conf]), 6) if value is not None else round(key_conf, 6)
        ),
        "key_confidence": round(key_conf, 6),
        "value_confidence": round(value_conf, 6) if value is not None else None,
    }


def _canonicalize_field_name(key_text: str, field_aliases: dict[str, str]) -> str:
    normalized = re.sub(r"[:\s]+", " ", key_text.strip()).lower().strip()
    if normalized in field_aliases:
        return field_aliases[normalized]
    return normalized.replace(" ", "_")


# ---------------------------------------------------------------------------
# Locale-aware number parsing
# ---------------------------------------------------------------------------


def parse_number(text: str, locale: str = LOCALE) -> float | None:
    """Parse a currency/quantity string into a float.

    Locale conventions for the thousands separator and decimal point:

    +--------+-----------------+----------------+
    | locale | thousands sep   | decimal point  |
    +========+=================+================+
    | id     | "." (1.591.600) | "," (600,50)   |
    | en     | "," (1,591,600) | "." (600.50)   |
    | auto   | heuristic       | heuristic      |
    +--------+-----------------+----------------+

    Returns None when the text holds no numeric content.
    """
    candidate = re.sub(r"[^0-9,.\-]", "", str(text).strip())
    if not candidate:
        return None

    if locale.lower() in {"id", "id-id", "in", "in-id"}:
        candidate = candidate.replace(".", "").replace(",", ".")
    elif locale.lower() in {"en", "en-us", "us", "en-in"}:
        candidate = candidate.replace(",", "")
    else:  # "auto" / anything unknown: resolve mixed or ambiguous separators
        candidate = _resolve_ambiguous_separators(candidate)

    try:
        return float(candidate)
    except ValueError:
        return None


def _resolve_ambiguous_separators(candidate: str) -> str:
    """Guess separator roles for locale='auto' ("1.591,60" vs "1,591.60")."""
    if "," in candidate and "." in candidate:
        if candidate.rfind(",") > candidate.rfind("."):
            return candidate.replace(".", "").replace(",", ".")  # 1.000,50
        return candidate.replace(",", "")  # 1,000.50
    if "." in candidate:
        return _disambiguate_single_separator(candidate, ".")
    if "," in candidate:
        return _disambiguate_single_separator(candidate, ",")
    return candidate


def _disambiguate_single_separator(candidate: str, separator: str) -> str:
    """Single-separator case: thousands groups are groups of exactly 3 digits."""
    unsigned = candidate.lstrip("-")
    parts = unsigned.split(separator)
    if len(parts) < 2:
        return candidate
    if not parts[0] or len(parts[0]) > 3:
        return candidate  # leading group >3 digits cannot be a decimal group
    if all(part.isdigit() and len(part) == 3 for part in parts[1:]):
        return candidate.replace(separator, "")  # thousands separator
    if all(part.isdigit() for part in parts[1:]) and len(parts) == 2:
        return candidate.replace(separator, ".")  # decimal point
    if len(parts) > 2:
        # e.g. "1.591.600" with unequal widths -> all but the last are thousands
        head = "".join(parts[:-1])
        return head + "." + parts[-1]
    return candidate.replace(separator, ".")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def _token_f1(
    true_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, float]:
    """Token-level micro precision/recall/F1 over the whole validation set."""
    true_tokens = [t for seq in true_sequences for t in seq]
    pred_tokens = [t for seq in pred_sequences for t in seq]

    tp = fp = fn = 0
    for true, pred in zip(true_tokens, pred_tokens, strict=False):
        if pred != "O":
            if pred == true:
                tp += 1
            else:
                fp += 1
        elif true != "O":
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_validation_set(
    examples: list[dict[str, Any]],
    *,
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    device: torch.device,
    confidence_threshold: float | None = None,
    locale: str = LOCALE,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Benchmark the model on a validation set.

    Args:
        examples: CORD-style records, each with:
            - ``image`` (PIL Image, path string, or ``image_path`` key)
            - ``words``: list of OCR word strings
            - ``boxes``: list of [x0, y0, x1, y1] per word
            - ``labels``: gold BIO labels per word (O/B-KEY/.../I-VALUE)
        batch_size: Documents per forward pass (larger is faster, needs VRAM).

    Returns:
        Dict with token-level and seqeval entity-level metrics, plus the mean
        confidence of non-O predictions.
    """
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    all_confidences: list[float] = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        images = [_resolve_example_image(example) for example in chunk]
        words_list = [list(example["words"]) for example in chunk]
        boxes_list = [
            [
                [int(c) for c in box]
                for box in example.get(
                    "boxes", [[0, 0, 0, 0]] * len(example["words"])
                )
            ]
            for example in chunk
        ]

        batch_results = predict_batch(
            images,
            words_list,
            boxes_list,
            model=model,
            processor=processor,
            device=device,
            confidence_threshold=confidence_threshold,
            locale=locale,
        )

        for example, result in zip(chunk, batch_results, strict=False):
            words = list(example["words"])
            gold = [str(label) for label in example["labels"]]

            pred_by_word: dict[int, dict[str, Any]] = {}
            for prediction in result["predictions"]:
                pred_by_word.setdefault(len(pred_by_word), prediction)
            pred_labels = [
                pred_by_word.get(idx, {"label": "O", "confidence": 0.0})["label"]
                for idx in range(len(words))
            ]

            true_sequences.append(gold)
            pred_sequences.append(pred_labels)
            all_confidences.extend(
                p["confidence"] for p in pred_by_word.values() if p["label"] != "O"
            )

    token_metrics = _token_f1(true_sequences, pred_sequences)

    try:
        from seqeval.metrics import classification_report, precision_score, recall_score
        from seqeval.metrics import f1_score as seqeval_f1

        entity_metrics = {
            "precision": float(
                precision_score(true_sequences, pred_sequences, zero_division=0)
            ),
            "recall": float(recall_score(true_sequences, pred_sequences, zero_division=0)),
            "f1": float(
                seqeval_f1(true_sequences, pred_sequences, average="micro", zero_division=0)
            ),
            "classification_report": classification_report(
                true_sequences, pred_sequences, zero_division=0
            ),
        }
    except ImportError:
        entity_metrics = {"f1": None, "classification_report": "seqeval not installed"}

    return {
        "num_examples": len(examples),
        "token_level": token_metrics,
        "entity_level": entity_metrics,
        "mean_non_o_confidence": fmean(all_confidences) if all_confidences else 0.0,
    }


def sweep_confidence_thresholds(
    examples: list[dict[str, Any]],
    *,
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    device: torch.device,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    locale: str = LOCALE,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Benchmark across confidence thresholds to pick the best trade-off."""
    results = []
    for threshold in thresholds:
        metrics = evaluate_validation_set(
            examples,
            model=model,
            processor=processor,
            device=device,
            confidence_threshold=threshold,
            locale=locale,
            batch_size=batch_size,
        )
        results.append(
            {
                "threshold": threshold,
                "token_f1": round(metrics["token_level"]["f1"], 4),
                "token_precision": round(metrics["token_level"]["precision"], 4),
                "token_recall": round(metrics["token_level"]["recall"], 4),
                "entity_f1": (
                    round(metrics["entity_level"]["f1"], 4)
                    if metrics["entity_level"].get("f1") is not None
                    else None
                ),
            }
        )
    return results


def _resolve_example_image(example: dict[str, Any]) -> Image.Image:
    """Resolve the image for a validation record (PIL, path, or blank fallback).

    Supports both an ``image`` field and the CORD-style ``image_path`` field.
    If the path is missing or unreadable, a blank white image is used so
    evaluation never hard-crashes (log the path to spot-check data quality).
    """
    image = example.get("image")
    if image is None:
        image = example.get("image_path")

    if isinstance(image, str | Path):
        path = Path(image).expanduser()
        if path.exists():
            return Image.open(path).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")

    return Image.new("RGB", (224, 224), color="white")


# ---------------------------------------------------------------------------
# Validation data loading (JSONL / JSON, CORD-style)
# ---------------------------------------------------------------------------


def load_validation_data(data_path: str | Path) -> list[dict[str, Any]]:
    """Load a CORD-style validation file into ``evaluate_validation_set`` format.

    Accepts JSONL (one JSON object per line) or a JSON array. Each record may
    contain any of:

    - ``image_path`` or ``image``: path to the receipt image (relative paths
      are resolved against the data file's directory).
    - ``words``: list of OCR word strings.
    - ``bboxes`` or ``boxes``: one ``[x0, y0, x1, y1]`` box per word.
    - ``ner_tags``, ``labels`` or ``bio_labels``: gold tags per word, either
      as BIO strings (``B-KEY``, ...) or label ids (0..4).

    Records without words are skipped. Returns a list of example dicts with
    normalized ``image``/``words``/``boxes``/``labels`` keys.
    """
    data_path = Path(data_path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {data_path}")

    base_dir = data_path.parent

    if data_path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with data_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        records = json.loads(data_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array or JSONL, got: {type(records).__name__}")

    examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        words = [str(w) for w in record.get("words") or []]
        if not words:
            continue

        raw_boxes = record.get("bboxes") or record.get("boxes") or [[0, 0, 0, 0]] * len(words)
        boxes = [[int(c) for c in box] for box in raw_boxes]

        raw_tags = record.get("ner_tags") or record.get("labels") or record.get("bio_labels") or []
        labels = [_resolve_label(tag) for tag in raw_tags]

        image_field = record.get("image_path") or record.get("image")
        if isinstance(image_field, str) and not Path(image_field).expanduser().is_absolute():
            image_field = str(base_dir / image_field)

        examples.append(
            {
                "image": image_field,
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "_source_index": index,
            }
        )

    return examples


def _resolve_label(tag: Any) -> str:
    """Coerce a gold tag to a BIO string (accepts ids and strings alike)."""
    if isinstance(tag, bool):
        return "O"
    if isinstance(tag, int):
        return ID2LABEL.get(tag, "O")
    label = str(tag)
    return label if label in LABEL2ID else "O"


# ---------------------------------------------------------------------------
# Optional: hyperparameters for continued fine-tuning (GPU required)
# ---------------------------------------------------------------------------


def suggested_training_hyperparameters() -> dict[str, Any]:
    """Suggested hyperparameters for continued fine-tuning of this checkpoint.

    These slot directly into ``LayoutLMv3Trainer`` from
    ``src/document_intelligence_engine/multimodal/training.py``; pass the
    fine-tuned checkpoint as ``model_name`` (instead of the base model) and
    point ``save_dir`` at a NEW folder so the original checkpoint is kept
    intact. CPU-only machines will be far too slow (LayoutLMv3 needs ~8 GB
    VRAM at batch size 4), so a GPU with fp16 is effectively required.
    """
    return {
        # Lower LR than the base-model default (5e-5): we are continuing
        # from an already-trained checkpoint, so large steps risk forgetting.
        "learning_rate": 2e-5,  # try 1e-5 if val F1 degrades vs. the frozen checkpoint
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,  # shorter warmup than base training (0.1)
        "num_epochs": 5,  # CORD is small; more than ~8 risks overfitting
        "batch_size": 4,  # 8 if VRAM allows (>= 12 GB)
        "gradient_accumulation_steps": 2,  # effective batch 8/16
        "max_length": MAX_SEQUENCE_LENGTH,
        "include_funsd": False,  # keep True only if you want FUNSD mixing
        # Training tips:
        # - Enable fp16 (set AMP in LayoutLMv3Trainer) for ~2x speedup.
        # - Freeze the visual backbone (layoutlmv3.visual) during the first
        #   epoch to stabilise training on small datasets.
        # - Keep the best checkpoint by validation seqeval F1 (already done by
        #   the trainer) and compare against this module's benchmark output.
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference_cord_finetuned",
        description="Inference + benchmarking for the fine-tuned CORD LayoutLMv3 checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("DRISE_MODEL_PATH", MODEL_PATH),
        help=f"Checkpoint folder (default: $DRISE_MODEL_PATH or {MODEL_PATH!r}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confidence threshold for inference (default: {CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--locale",
        default=LOCALE,
        choices=("id", "en", "auto"),
        help="Number parsing locale (default: id).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cpu' or 'cuda:0' (default: auto).",
    )

    mode = parser.add_argument_group("modes").add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Benchmark on a validation file and print F1 metrics.",
    )
    mode.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Sweep confidence thresholds on a validation file.",
    )
    mode.add_argument(
        "--image",
        help="Run inference on a single receipt image.",
    )

    parser.add_argument(
        "--data-path",
        help="Path to a validation JSONL/JSON file (for --evaluate/--tune-threshold).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Documents per forward pass (default: 8).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds to sweep (for --tune-threshold).",
    )
    parser.add_argument(
        "--words",
        help="JSON array of OCR words, e.g. '[\"Grand\",\"Total\"]' (for --image).",
    )
    parser.add_argument(
        "--boxes",
        help="JSON array of [x0,y0,x1,y1] boxes, e.g. '[[0,0,100,20]]' (for --image).",
    )
    return parser


def _run_evaluate(args: argparse.Namespace) -> int:
    if not args.data_path:
        print("--evaluate requires --data-path")
        return 2
    examples = load_validation_data(args.data_path)
    if not examples:
        print("No valid examples found in", args.data_path)
        return 2

    model, processor, device = load_model(args.model_path, device=args.device)
    print(f"Loaded model on {device} — {len(examples)} examples\n")

    metrics = evaluate_validation_set(
        examples,
        model=model,
        processor=processor,
        device=device,
        confidence_threshold=args.threshold,
        locale=args.locale,
        batch_size=args.batch_size,
    )
    token = metrics["token_level"]
    entity = metrics["entity_level"]
    print(f"Examples evaluated : {metrics['num_examples']}")
    print(f"Token-level F1     : {token['f1']:.4f} "
          f"(P {token['precision']:.4f} / R {token['recall']:.4f})")
    if entity.get("f1") is not None:
        print(f"Entity-level F1    : {entity['f1']:.4f} "
              f"(P {entity['precision']:.4f} / R {entity['recall']:.4f})")
    print(f"Mean non-O conf    : {metrics['mean_non_o_confidence']:.4f}")
    if entity.get("classification_report"):
        print("\n--- seqeval classification report ---\n")
        print(entity["classification_report"])
    return 0


def _run_tune_threshold(args: argparse.Namespace) -> int:
    if not args.data_path:
        print("--tune-threshold requires --data-path")
        return 2
    examples = load_validation_data(args.data_path)
    if not examples:
        print("No valid examples found in", args.data_path)
        return 2

    model, processor, device = load_model(args.model_path, device=args.device)
    print(f"Loaded model on {device} — {len(examples)} examples\n")

    rows = sweep_confidence_thresholds(
        examples,
        model=model,
        processor=processor,
        device=device,
        thresholds=tuple(args.thresholds),
        locale=args.locale,
        batch_size=args.batch_size,
    )

    print(f"{'threshold':>10} | {'token_f1':>8} | {'entity_f1':>9}")
    print("-" * 34)
    for row in rows:
        entity = f"{row['entity_f1']:.4f}" if row["entity_f1"] is not None else "   n/a"
        print(f"{row['threshold']:>10.3f} | {row['token_f1']:>8.4f} | {entity:>9}")

    # Recommend the threshold that maximizes entity F1 (token F1 as a fallback).
    best = max(
        rows,
        key=lambda r: (r["entity_f1"] is not None, r["entity_f1"] or -1, r["token_f1"]),
    )
    print(f"\nRecommended CONFIDENCE_THRESHOLD: {best['threshold']}")
    return 0


def _run_image(args: argparse.Namespace) -> int:
    if not args.words or not args.boxes:
        print("--image requires --words and --boxes (JSON arrays from your OCR pass).")
        return 2
    try:
        words = json.loads(args.words)
        boxes = json.loads(args.boxes)
    except json.JSONDecodeError as exc:
        print(f"Could not parse --words/--boxes as JSON: {exc}")
        return 2

    model, processor, device = load_model(args.model_path, device=args.device)
    image = Image.open(args.image).convert("RGB")
    result = predict_receipt(
        image,
        words,
        boxes,
        model=model,
        processor=processor,
        device=device,
        confidence_threshold=args.threshold,
        locale=args.locale,
    )

    print(json.dumps(result["key_value_pairs"], indent=2))
    if result["errors"]:
        print("\nGrouping warnings:", json.dumps(result["errors"], indent=2))
    return 0


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()

    if _args.evaluate:
        raise SystemExit(_run_evaluate(_args))
    if _args.tune_threshold:
        raise SystemExit(_run_tune_threshold(_args))
    if _args.image:
        raise SystemExit(_run_image(_args))

    # No mode selected: show help.
    _build_arg_parser().print_help()