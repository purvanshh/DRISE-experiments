"""Convert FUNSD and CORD Hugging Face datasets into experiment annotations."""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
FUNSD_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "hf_datasets" / "funsd"
CORD_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "hf_datasets" / "cord"
SEED = 42

DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
]
AMOUNT_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?(?!\d)")

INVOICE_KEYWORDS = ("invoice", "invoice no", "invoice #", "inv", "receipt", "receipt no", "receipt #", "bill no", "document no")
DATE_KEYWORDS = ("date", "invoice date", "receipt date", "bill date")
VENDOR_KEYWORDS = ("vendor", "seller", "supplier", "merchant", "store", "company", "from")
TOTAL_KEYWORDS = ("total", "amount", "amount due", "grand total", "total due", "balance")


@dataclass
class Span:
    label: str
    text: str
    start: int
    end: int


def main() -> None:
    random.seed(SEED)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    FUNSD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CORD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    funsd = load_dataset("nielsr/funsd")
    cord = load_dataset("katanaml/cord")

    records = []
    records.extend(_convert_funsd_dataset(funsd))
    records.extend(_convert_cord_dataset(cord))

    random.Random(SEED).shuffle(records)
    train_records, val_records, test_records = _split_records(records)

    _write_jsonl(ANNOTATIONS_DIR / "train.jsonl", train_records)
    _write_jsonl(ANNOTATIONS_DIR / "val.jsonl", val_records)
    _write_jsonl(ANNOTATIONS_DIR / "test.jsonl", test_records)

    summary = {
        "total_documents": len(records),
        "train_documents": len(train_records),
        "val_documents": len(val_records),
        "test_documents": len(test_records),
        "sources": {
            "funsd": len([record for record in records if record["metadata"]["source_dataset"] == "funsd"]),
            "cord": len([record for record in records if record["metadata"]["source_dataset"] == "cord"]),
        },
    }
    print(json.dumps(summary, indent=2))


def _convert_funsd_dataset(dataset: DatasetDict) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name, split_dataset in dataset.items():
        records.extend(_convert_funsd_split(split_dataset, split_name))
    return records


def _convert_funsd_split(split_dataset: Dataset, split_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    label_names = split_dataset.features["ner_tags"].feature.names
    for row in split_dataset:
        doc_id = f"funsd_{split_name}_{row['id']}"
        image_path = FUNSD_IMAGE_DIR / f"{doc_id}.png"
        if not image_path.exists():
            row["image"].save(image_path)

        words = [str(word).strip() for word in row["words"]]
        bboxes = row["bboxes"]
        tags = [label_names[index] for index in row["ner_tags"]]
        ocr_tokens = _build_tokens(words, bboxes)
        spans = _group_bio_spans(words, tags)
        question_answer_map = _pair_questions_with_answers(spans)
        ocr_text = _join_ocr_text(words)
        ground_truth = _extract_funsd_ground_truth(question_answer_map, ocr_text)

        records.append(
            {
                "doc_id": doc_id,
                "image_path": str(image_path.resolve()),
                "ocr_text": ocr_text,
                "ocr_tokens": ocr_tokens,
                "ground_truth": ground_truth,
                "metadata": {
                    "source_dataset": "funsd",
                    "source_split": split_name,
                    "source_id": row["id"],
                },
            }
        )
    return records


def _convert_cord_dataset(dataset: DatasetDict) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name, split_dataset in dataset.items():
        records.extend(_convert_cord_split(split_dataset, split_name))
    return records


def _convert_cord_split(split_dataset: Dataset, split_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    label_names = split_dataset.features["ner_tags"].feature.names
    for row in split_dataset:
        doc_id = f"cord_{split_name}_{row['id']}"
        words = [str(word).strip() for word in row["words"]]
        bboxes = row["bboxes"]
        source_image_path = Path(row["image_path"]).resolve()
        image_path = CORD_IMAGE_DIR / f"{doc_id}{source_image_path.suffix.lower() or '.png'}"
        if not image_path.exists():
            if source_image_path.exists():
                shutil.copy2(source_image_path, image_path)
            else:
                _render_cord_page_image(words=words, bboxes=bboxes, output_path=image_path)
        tags = [label_names[index] for index in row["ner_tags"]]
        ocr_tokens = _build_tokens(words, bboxes)
        ocr_text = _join_ocr_text(words)
        ground_truth = _extract_cord_ground_truth(words, tags, ocr_text)

        records.append(
            {
                "doc_id": doc_id,
                "image_path": str(image_path.resolve()),
                "ocr_text": ocr_text,
                "ocr_tokens": ocr_tokens,
                "ground_truth": ground_truth,
                "metadata": {
                    "source_dataset": "cord",
                    "source_split": split_name,
                    "source_id": row["id"],
                },
            }
        )
    return records


def _build_tokens(words: list[str], bboxes: list[list[int]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for word, bbox in zip(words, bboxes, strict=False):
        cleaned = word.strip()
        if not cleaned:
            continue
        tokens.append(
            {
                "text": cleaned,
                "bbox": [int(value) for value in bbox],
                "confidence": 1.0,
                "page_number": 1,
            }
        )
    return tokens


def _render_cord_page_image(words: list[str], bboxes: list[list[int]], output_path: Path) -> None:
    canvas_size = 1200
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for word, bbox in zip(words, bboxes, strict=False):
        if not word:
            continue
        x0, y0, x1, y1 = [max(0, min(canvas_size - 1, int(value * 1.1))) for value in bbox]
        if x1 <= x0:
            x1 = min(canvas_size - 1, x0 + 40)
        if y1 <= y0:
            y1 = min(canvas_size - 1, y0 + 20)
        draw.rectangle([x0, y0, x1, y1], outline="lightgray", width=1)
        draw.text((x0 + 2, y0 + 1), str(word), fill="black", font=font)

    image.save(output_path)


def _group_bio_spans(words: list[str], tags: list[str]) -> list[Span]:
    spans: list[Span] = []
    current_label: str | None = None
    current_words: list[str] = []
    start_index = 0

    for index, (word, tag) in enumerate(zip(words, tags, strict=False)):
        if tag == "O":
            if current_label is not None and current_words:
                spans.append(
                    Span(label=current_label, text=" ".join(current_words).strip(), start=start_index, end=index - 1)
                )
            current_label = None
            current_words = []
            continue

        prefix, base_label = tag.split("-", maxsplit=1)
        if prefix == "B" or current_label != base_label:
            if current_label is not None and current_words:
                spans.append(
                    Span(label=current_label, text=" ".join(current_words).strip(), start=start_index, end=index - 1)
                )
            current_label = base_label
            current_words = [word]
            start_index = index
        else:
            current_words.append(word)

    if current_label is not None and current_words:
        spans.append(Span(label=current_label, text=" ".join(current_words).strip(), start=start_index, end=len(words) - 1))
    return spans


def _pair_questions_with_answers(spans: list[Span]) -> list[tuple[str, str]]:
    questions = [span for span in spans if span.label == "QUESTION" and span.text]
    answers = [span for span in spans if span.label == "ANSWER" and span.text]
    pairs: list[tuple[str, str]] = []
    used_answers: set[int] = set()

    for question in questions:
        best_index: int | None = None
        best_distance: tuple[int, int] | None = None
        for index, answer in enumerate(answers):
            if index in used_answers:
                continue
            distance = abs(answer.start - question.end)
            distance_key = (distance, answer.start)
            if best_distance is None or distance_key < best_distance:
                best_distance = distance_key
                best_index = index
        if best_index is not None:
            used_answers.add(best_index)
            pairs.append((question.text, answers[best_index].text))
    return pairs


def _extract_funsd_ground_truth(question_answer_pairs: list[tuple[str, str]], ocr_text: str) -> dict[str, Any]:
    invoice_number = ""
    date_value = ""
    vendor = ""
    total_amount: float | None = None

    for question, answer in question_answer_pairs:
        lowered = question.lower()
        if not invoice_number and _contains_keyword(lowered, INVOICE_KEYWORDS):
            invoice_number = answer.strip()
            continue
        if not date_value and _contains_keyword(lowered, DATE_KEYWORDS):
            date_value = _normalize_date(answer)
            continue
        if not vendor and _contains_keyword(lowered, VENDOR_KEYWORDS):
            vendor = answer.strip()
            continue
        if total_amount is None and _contains_keyword(lowered, TOTAL_KEYWORDS):
            total_amount = _parse_amount(answer)

    if not date_value:
        date_value = _search_date(ocr_text)

    if total_amount is None:
        total_amount = _search_amount_from_text(ocr_text, prefer_last=True)

    return {
        "invoice_number": invoice_number,
        "date": date_value,
        "vendor": vendor,
        "total_amount": total_amount,
        "line_items": [],
    }


def _extract_cord_ground_truth(words: list[str], tags: list[str], ocr_text: str) -> dict[str, Any]:
    grouped = _group_tagged_text(tags, words)
    vendor = ""

    total_amount = _first_amount(
        grouped,
        [
            "I-total.total_price",
            "I-total.cashprice",
            "I-total.creditcardprice",
            "I-total.emoneyprice",
            "I-sub_total.subtotal_price",
        ],
    )
    if total_amount is None:
        total_amount = _search_amount_from_text(ocr_text, prefer_last=True)

    date_value = _search_date(ocr_text)
    invoice_number = _search_invoice_number(ocr_text)
    line_items = _extract_cord_line_items(grouped)

    return {
        "invoice_number": invoice_number,
        "date": date_value,
        "vendor": vendor,
        "total_amount": total_amount,
        "line_items": line_items,
    }


def _group_tagged_text(tags: list[str], words: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    current_label: str | None = None
    current_words: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_words
        if current_label and current_words:
            grouped[current_label].append(" ".join(current_words).strip())
        current_label = None
        current_words = []

    for tag, word in zip(tags, words, strict=False):
        if tag == "O":
            flush()
            continue
        if current_label == tag:
            current_words.append(word)
        else:
            flush()
            current_label = tag
            current_words = [word]
    flush()
    return grouped


def _extract_cord_line_items(grouped: dict[str, list[str]]) -> list[dict[str, Any]]:
    descriptions = grouped.get("I-menu.nm", []) + grouped.get("I-menu.sub_nm", [])
    quantities = grouped.get("I-menu.cnt", []) + grouped.get("I-menu.sub_cnt", [])
    unit_prices = grouped.get("I-menu.unitprice", []) + grouped.get("I-menu.price", []) + grouped.get("I-menu.sub_price", [])

    max_len = max(len(descriptions), len(quantities), len(unit_prices), 0)
    line_items: list[dict[str, Any]] = []
    for index in range(max_len):
        description = descriptions[index].strip() if index < len(descriptions) else ""
        quantity = _parse_number(quantities[index]) if index < len(quantities) else None
        unit_price = _parse_amount(unit_prices[index]) if index < len(unit_prices) else None
        if not description and quantity is None and unit_price is None:
            continue
        line_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
            }
        )
    return line_items


def _first_amount(grouped: dict[str, list[str]], labels: Iterable[str]) -> float | None:
    for label in labels:
        for value in grouped.get(label, []):
            amount = _parse_amount(value)
            if amount is not None:
                return amount
    return None


def _search_date(text: str) -> str:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize_date(match.group(0))
    return ""


def _normalize_date(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    normalized = cleaned.replace("/", "-")
    parts = normalized.split("-")
    if len(parts) != 3:
        return cleaned

    if len(parts[0]) == 4:
        year, month, day = parts
    else:
        month, day, year = parts
        if len(year) == 2:
            year = f"20{year}"

    try:
        month_int = int(month)
        day_int = int(day)
        year_int = int(year)
    except ValueError:
        return cleaned

    if not (1 <= month_int <= 12 and 1 <= day_int <= 31 and year_int > 0):
        return cleaned
    return f"{year_int:04d}-{month_int:02d}-{day_int:02d}"


def _search_invoice_number(text: str) -> str:
    pattern = re.compile(
        r"(?:invoice|receipt|bill)\s*(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]*)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _search_amount_from_text(text: str, prefer_last: bool) -> float | None:
    candidates = [match.group(0) for match in AMOUNT_PATTERN.finditer(text)]
    if not candidates:
        return None
    source = candidates[-1] if prefer_last else candidates[0]
    return _parse_amount(source)


def _parse_amount(value: str) -> float | None:
    if not value:
        return None
    matches = AMOUNT_PATTERN.findall(str(value))
    if not matches:
        return None
    candidate = matches[-1].replace(",", "")
    try:
        return float(candidate)
    except ValueError:
        return None


def _parse_number(value: str) -> float | int | None:
    amount = _parse_amount(value)
    if amount is None:
        return None
    if amount.is_integer():
        return int(amount)
    return amount


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _join_ocr_text(words: list[str]) -> str:
    return " ".join(word for word in words if word).strip()


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(records)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]
    return train_records, val_records, test_records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
