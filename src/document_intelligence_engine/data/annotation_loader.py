"""Annotation loading helpers for experiment datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from document_intelligence_engine.domain.experiment_models import ProcessedDocument


def load_annotations(dataset_path: str | Path) -> list[ProcessedDocument]:
    """Load experiment annotations from JSON or JSONL."""

    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "samples" in payload:
            records = payload["samples"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported dataset structure in {path}")

    return [_coerce_document(record, base_dir=path.parent) for record in records]


def _coerce_document(record: dict[str, Any], base_dir: Path) -> ProcessedDocument:
    image_path = str(record.get("image_path", ""))
    if image_path and not Path(image_path).is_absolute():
        image_path = str((base_dir / image_path).resolve())

    ocr_tokens = list(record.get("ocr_tokens", []))
    ocr_text = str(record.get("ocr_text", "")).strip()
    if not ocr_text and ocr_tokens:
        ocr_text = "\n".join(str(token.get("text", "")).strip() for token in ocr_tokens if token.get("text"))

    document: ProcessedDocument = {
        "doc_id": str(record.get("doc_id") or record.get("id") or _derive_doc_id(image_path, ocr_text)),
        "image_path": image_path,
        "ocr_text": ocr_text,
        "ocr_tokens": ocr_tokens,
        "ground_truth": _normalize_ground_truth(record.get("ground_truth")),
    }
    return document


def _derive_doc_id(image_path: str, ocr_text: str) -> str:
    if image_path:
        return Path(image_path).stem
    if ocr_text:
        return ocr_text[:24].replace(" ", "_")
    return "document"


def _normalize_ground_truth(ground_truth: Any) -> dict[str, Any] | None:
    if ground_truth is None:
        return None
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must be a JSON object when provided.")

    normalized = dict(ground_truth)
    normalized.setdefault("invoice_number", "")
    normalized.setdefault("date", "")
    normalized.setdefault("vendor", "")
    normalized.setdefault("total_amount", None)
    normalized.setdefault("line_items", [])

    if normalized["line_items"] is None:
        normalized["line_items"] = []
    if not isinstance(normalized["line_items"], list):
        raise ValueError("ground_truth.line_items must be a list.")

    return normalized
