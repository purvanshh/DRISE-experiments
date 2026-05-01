"""Prepare shared OCR annotations for experiment datasets.

This script ensures every annotation record contains:
- ``ocr_text``
- ``ocr_tokens``
- ``ocr_metadata``

By default it reuses existing OCR stored in the JSONL files and only invokes
the repo OCR pipeline when OCR is missing or ``--force-reextract`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from ingestion.pipeline import process_document_with_metadata


DEFAULT_DATASETS = (
    "data/annotations/train.jsonl",
    "data/annotations/val.jsonl",
    "data/annotations/test.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared OCR fields for experiment annotations.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Annotation JSONL files to update.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Ignore existing OCR fields and rerun the ingestion OCR pipeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record limit per dataset for smoke testing.",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for dataset_path in args.datasets:
        summaries.append(
            _prepare_dataset(
                Path(dataset_path).expanduser().resolve(),
                force_reextract=args.force_reextract,
                limit=args.limit,
            )
        )
    print(json.dumps({"datasets": summaries}, indent=2))


def _prepare_dataset(dataset_path: Path, *, force_reextract: bool, limit: int | None) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    documents = load_annotations(dataset_path)
    raw_records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    updated_records: list[dict[str, Any]] = []
    reused_count = 0
    regenerated_count = 0

    for index, (raw_record, document) in enumerate(zip(raw_records, documents, strict=False)):
        if limit is not None and index >= limit:
            updated_records.append(raw_record)
            continue

        image_path = str(document.get("image_path", ""))
        has_cached_ocr = bool(document.get("ocr_text")) and bool(document.get("ocr_tokens"))
        if has_cached_ocr and not force_reextract:
            updated_record = dict(raw_record)
            updated_record["raw_image_path"] = str(document.get("raw_image_path") or image_path)
            updated_record["ocr_text"] = str(document.get("ocr_text", ""))
            updated_record["ocr_tokens"] = list(document.get("ocr_tokens", []))
            updated_record["ocr_metadata"] = {
                "source": "cached_annotation",
                "engine": "dataset_annotation",
                "language": "en",
                "page_count": int(document.get("ocr_metadata", {}).get("page_count", 1) or 1),
                "token_count": len(document.get("ocr_tokens", [])),
                "timing": dict(document.get("ocr_metadata", {}).get("timing", {})),
                "reused_cached_ocr": True,
            }
            reused_count += 1
        else:
            if not image_path:
                raise ValueError(f"Document {document.get('doc_id')} is missing image_path; cannot recompute OCR.")
            ocr_payload = process_document_with_metadata(image_path, debug=False)
            ocr_tokens = [_normalize_token(token) for token in ocr_payload["ocr_tokens"]]
            ocr_text = _ocr_text_from_tokens(ocr_tokens)

            updated_record = dict(raw_record)
            updated_record["raw_image_path"] = str(document.get("raw_image_path") or image_path)
            updated_record["ocr_text"] = ocr_text
            updated_record["ocr_tokens"] = ocr_tokens
            updated_record["ocr_metadata"] = {
                "source": "ingestion_pipeline",
                "engine": "paddleocr",
                "language": "en",
                "page_count": int(ocr_payload.get("page_count", 1)),
                "token_count": len(ocr_tokens),
                "timing": dict(ocr_payload.get("timing", {})),
                "reused_cached_ocr": False,
            }
            regenerated_count += 1

        updated_records.append(updated_record)

    dataset_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in updated_records) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(dataset_path),
        "record_count": len(updated_records),
        "reused_cached_ocr": reused_count,
        "regenerated_ocr": regenerated_count,
        "limit": limit,
    }


def _normalize_token(token: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(token)
    normalized.setdefault("page_number", 1)
    return normalized


def _ocr_text_from_tokens(tokens: list[dict[str, Any]]) -> str:
    if not tokens:
        return ""
    ordered_tokens = sorted(
        tokens,
        key=lambda item: (
            int(item.get("page_number", 1)),
            int(item.get("bbox", [0, 0, 0, 0])[1]),
            int(item.get("bbox", [0, 0, 0, 0])[0]),
        ),
    )
    return "\n".join(str(token.get("text", "")).strip() for token in ordered_tokens if token.get("text"))


if __name__ == "__main__":
    main()
