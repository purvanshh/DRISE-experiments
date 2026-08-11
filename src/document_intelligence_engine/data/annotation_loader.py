"""Annotation loading helpers for experiment datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.experiment_models import ProcessedDocument


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
    image_path = _resolve_document_path(str(record.get("image_path", "")), base_dir)
    raw_image_path = _resolve_document_path(str(record.get("raw_image_path", image_path or "")), base_dir)

    ocr_tokens = list(record.get("ocr_tokens", []))
    ocr_text = str(record.get("ocr_text", "")).strip()
    if not ocr_text and ocr_tokens:
        ocr_text = "\n".join(str(token.get("text", "")).strip() for token in ocr_tokens if token.get("text"))

    document: ProcessedDocument = {
        "doc_id": str(record.get("doc_id") or record.get("id") or _derive_doc_id(image_path, ocr_text)),
        "image_path": image_path,
        "raw_image_path": raw_image_path or image_path,
        "ocr_text": ocr_text,
        "ocr_tokens": ocr_tokens,
        "ocr_metadata": _normalize_ocr_metadata(record.get("ocr_metadata"), token_count=len(ocr_tokens)),
        "ground_truth": _normalize_ground_truth(record.get("ground_truth")),
    }
    return document


def _resolve_document_path(path_value: str, base_dir: Path) -> str:
    if not path_value:
        return ""

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        return str((base_dir / candidate).resolve())
    if candidate.exists():
        return str(candidate)

    repo_root = base_dir.parents[1] if len(base_dir.parents) >= 2 else base_dir
    parts = list(candidate.parts)
    for marker in ("data", repo_root.name):
        if marker not in parts:
            continue
        suffix = parts[parts.index(marker) :]
        rebased = repo_root.joinpath(*suffix) if marker != repo_root.name else repo_root.joinpath(*suffix[1:])
        if rebased.exists():
            return str(rebased.resolve())
    return str(candidate)


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

    normalized["line_items"] = _clean_ground_truth_line_items(normalized["line_items"])

    return normalized


def _clean_ground_truth_line_items(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove annotation artifacts introduced by the dataset conversion.

    The CORD conversion zips ``description``/``quantity``/``unit_price`` groups
    by index, which produces duplicate rows with an empty description (e.g. a
    second row carrying only the quantity or the grand-total price). These
    artifacts make exact line-item matching impossible and are merged back into
    the described row or dropped.
    """
    cleaned: list[dict[str, Any]] = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        quantity = item.get("quantity")
        unit_price = item.get("unit_price")

        # The CORD conversion emits unit_price=0.0 for items where the price
        # was not captured cleanly. A zero price is never a real unit price, so
        # treat it as unknown rather than scoring it as a literal "0".
        if unit_price == 0 and description:
            unit_price = None
            item = {**item, "unit_price": None}

        if not description and quantity is None and unit_price in (None, ""):
            continue

        if not description and unit_price not in (None, ""):
            # An orphan row that duplicates the previous described row's price
            # (e.g. a quantity or a totals row). Fold its quantity into the
            # described row when the price matches, otherwise drop it.
            if cleaned and cleaned[-1].get("unit_price") == unit_price:
                target = cleaned[-1]
                if target.get("quantity") in (None, "") and quantity not in (None, ""):
                    target["quantity"] = quantity
                continue
            if quantity in (None, "") or unit_price is None:
                continue

        cleaned.append(dict(item))

    return cleaned


def _normalize_ocr_metadata(ocr_metadata: Any, *, token_count: int) -> dict[str, Any]:
    if not isinstance(ocr_metadata, dict):
        return {
            "source": "annotation",
            "engine": "unknown",
            "language": "en",
            "page_count": 1,
            "token_count": token_count,
            "reused_cached_ocr": True,
        }

    normalized = dict(ocr_metadata)
    normalized.setdefault("source", "annotation")
    normalized.setdefault("engine", "unknown")
    normalized.setdefault("language", "en")
    normalized.setdefault("page_count", 1)
    normalized.setdefault("token_count", token_count)
    normalized.setdefault("reused_cached_ocr", True)
    return normalized
