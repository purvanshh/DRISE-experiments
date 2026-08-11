"""Evaluate DRISE pipeline outputs against ground truth.

Computes masked micro-F1 (only fields present in ground truth), document
exact-match, per-field F1, and conditional per-field F1.

Usage:
    python scripts/eval_drise.py results.json   # evaluate an existing results file
    python scripts/eval_drise.py --run          # run the DRISE pipeline live
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_intelligence_engine.data.annotation_loader import load_annotations  # noqa: E402

FIELDS = ["invoice_number", "date", "vendor", "total_amount", "line_items"]


def tokens(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                for key in sorted(item):
                    out += tokens(item[key])
            else:
                out += tokens(item)
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out += tokens(value[key])
        return out
    return re.findall(r"[a-z0-9.]+", str(value).lower())


def norm_amt(value):
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return round(float(value), 2)
    cleaned = re.sub(r"[^0-9.\-]+", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def norm_text(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text or None


def norm_date(value):
    if value in (None, ""):
        return None
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", str(value))
    return match.group(1) if match else norm_text(value)


def normalize_field(value, field_name):
    if field_name == "line_items":
        return sorted(
            (normalize_line_item(item) for item in (value or [])),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if field_name == "total_amount":
        amount = norm_amt(value)
        if amount is None:
            return None
        return f"{amount:.2f}"
    if field_name == "date":
        return norm_date(value)
    return norm_text(value)


def normalize_line_item(item):
    if not isinstance(item, dict):
        return norm_text(item)
    return {key: normalize_field(item.get(key), key) for key in sorted(item)}


def doc_exact(pred, gt, fields):
    return all(normalize_field(pred.get(f), f) == normalize_field(gt.get(f), f) for f in fields)


def evaluate(preds_by_doc, annotations):
    tp = {f: 0 for f in FIELDS}
    fp = {f: 0 for f in FIELDS}
    fn = {f: 0 for f in FIELDS}
    gt_present = {f: 0 for f in FIELDS}
    exact_all = 0
    exact_no_li = 0
    li_exact = 0
    li_gt = 0

    for ann in annotations:
        gt = ann["ground_truth"]
        pred = preds_by_doc.get(str(ann["doc_id"]), {}).get("extracted_fields", {})
        if doc_exact(pred, gt, FIELDS):
            exact_all += 1
        if doc_exact(pred, gt, FIELDS[:4]):
            exact_no_li += 1
        if gt.get("line_items"):
            li_gt += 1
            if normalize_field(pred.get("line_items"), "line_items") == normalize_field(
                gt.get("line_items"), "line_items"
            ):
                li_exact += 1

        for f in FIELDS:
            gtv = gt.get(f)
            pv = pred.get(f)
            if gtv in (None, "", [], {}):
                continue
            gt_present[f] += 1
            gt_tokens = Counter(tokens(gtv))
            pred_tokens = Counter(tokens(pv))
            inter = gt_tokens & pred_tokens
            hit = sum(inter.values())
            tp[f] += hit
            fn[f] += sum(gt_tokens.values()) - hit
            fp[f] += sum(pred_tokens.values()) - hit

    print("=== Metrics ===")
    total_tp = sum(tp.values())
    total_p = total_tp + sum(fp.values())
    total_g = total_tp + sum(fn.values())
    precision = total_tp / total_p if total_p else 0.0
    recall = total_tp / total_g if total_g else 0.0
    masked_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"masked micro-F1 (fields present in GT): {masked_f1:.4f}")
    print(f"  (P={precision:.4f} R={recall:.4f})")
    print(f"exact match (5 fields): {exact_all}/{len(annotations)} "
          f"= {exact_all / len(annotations):.4f}")
    print(f"exact match (without line_items): {exact_no_li}/{len(annotations)}")
    print(f"line_items exact: {li_exact}/{li_gt}")
    print()
    print("=== Per-field token F1 (GT present only) ===")
    for f in FIELDS:
        p = tp[f] / (tp[f] + fp[f]) if (tp[f] + fp[f]) else 0.0
        r = tp[f] / (tp[f] + fn[f]) if (tp[f] + fn[f]) else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{f}: GT-present={gt_present[f]}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
    return masked_f1, exact_all / len(annotations)


def load_results(path):
    payload = json.loads(Path(path).read_text())
    return {str(record["doc_id"]): record.get("output", record) for record in payload}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help="Existing results JSON to evaluate")
    args = parser.parse_args()

    annotations = load_annotations(ROOT / "data/annotations/test.jsonl")

    if args.results:
        preds = load_results(args.results)
    else:
        from document_intelligence_engine.pipelines.drise import DRISEPipeline

        pipeline = DRISEPipeline({"use_layout": True, "use_constraints": True})
        preds = {}
        for ann in annotations:
            output = pipeline.run(ann)
            preds[str(ann["doc_id"])] = output
            if len(preds) % 50 == 0:
                print(f"  processed {len(preds)}/201", file=sys.stderr)

    evaluate(preds, annotations)


if __name__ == "__main__":
    main()
