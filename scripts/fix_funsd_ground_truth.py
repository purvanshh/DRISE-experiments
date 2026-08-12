"""Apply the FUNSD ground-truth cleanup to existing annotation files.

The dataset converter (`scripts/convert_datasets.py`) forces FUNSD forms into
the invoice schema, which produces unreliable labels: totals that are years or
credit-card numbers, arbitrary form text as vendors, and phantom invoice
numbers. This script applies the same validation rules the converter now uses
to the already-generated `data/annotations/*.jsonl` files so the held-out
evaluation set is cleaned without perturbing document ids or the train/val/test
split.

Rules (FUNSD records only, detected via ``metadata.source_dataset``):
  - total_amount: set to null unless it is a plausible invoice amount
  - vendor:      set to "" unless it looks like a company name
  - invoice_number: set to "" (forms do not contain invoice numbers)
  - date:        set to "" unless it normalizes to a valid ISO date

Usage:
    python scripts/fix_funsd_ground_truth.py [--annotations data/annotations]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from convert_datasets import (  # noqa: E402
    _is_plausible_total_amount,
    _is_valid_iso_date,
    _looks_like_vendor,
    _normalize_date,
)


def _clean_funsd_ground_truth(ground_truth: dict) -> dict:
    cleaned = dict(ground_truth)
    cleaned["invoice_number"] = ""
    vendor = str(cleaned.get("vendor") or "").strip()
    cleaned["vendor"] = vendor if _looks_like_vendor(vendor) else ""

    total_amount = cleaned.get("total_amount")
    if not _is_plausible_total_amount(float(total_amount) if total_amount is not None else None):
        cleaned["total_amount"] = None

    date_value = str(cleaned.get("date") or "").strip()
    normalized_date = _normalize_date(date_value) if date_value else ""
    cleaned["date"] = normalized_date if _is_valid_iso_date(normalized_date) else ""

    return cleaned


def _clean_file(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    changed = 0
    for record in records:
        if record.get("metadata", {}).get("source_dataset") != "funsd":
            continue
        ground_truth = record.get("ground_truth")
        if not isinstance(ground_truth, dict):
            continue
        cleaned = _clean_funsd_ground_truth(ground_truth)
        if cleaned != ground_truth:
            record["ground_truth"] = cleaned
            changed += 1
    path.write_text("\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n", encoding="utf-8")
    return len(records), changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations", help="Directory of JSONL annotation files")
    args = parser.parse_args()

    target_dir = Path(args.annotations).expanduser().resolve()
    for path in sorted(target_dir.glob("*.jsonl")):
        total, changed = _clean_file(path)
        print(f"{path.name}: {total} records, {changed} FUNSD ground-truth records updated")


if __name__ == "__main__":
    main()
