"""Benchmark the fine-tuned CORD checkpoint on the katanaml/cord test split.

Token-level and seqeval entity-level F1, plus a confidence-threshold sweep.

Usage:
    python scripts/benchmark_cord_finetuned.py [--model-path PATH] \
        [--split test] [--batch-size 8] [--tune]

This mirrors the training setup in ``Drise Notebook.ipynb``:
  - Data source: the cached ``katanaml/cord`` dataset (``words``,
    ``bboxes``, ``ner_tags``, ``image_path``). Source images are usually not
    available, so documents are rendered from tokens exactly like
    ``cord_dataset._render_image_from_tokens`` during training.
  - Labels: katanaml ``ner_tags`` mapped to the project 5-class BIO scheme
    (``O`` / ``B-KEY`` / ``I-KEY`` / ``B-VALUE`` / ``I-VALUE``). CORD only
    annotates values, so non-O gold labels are VALUE spans; KEY supervision
    comes from the FUNSD mix during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence_engine.multimodal.cord_dataset import (  # noqa: E402
    _cord_label_to_bio,
    _render_image_from_tokens,
)
from inference_cord_finetuned import (  # noqa: E402
    evaluate_validation_set,
    load_model,
    sweep_confidence_thresholds,
)


def _build_examples(split: str) -> list[dict[str, object]]:
    dataset = load_dataset("katanaml/cord")
    label_names = dataset[split].features["ner_tags"].feature.names
    examples = []
    for record in dataset[split]:
        words = [str(w) for w in record["words"]]
        boxes = [[int(c) for c in b] for b in record["bboxes"]]
        if not words:
            continue
        labels = []
        previous = "O"
        for tag_id in record["ner_tags"]:
            tag = label_names[tag_id] if isinstance(tag_id, int) else str(tag_id)
            is_first = not labels or previous == "O"
            label = _cord_label_to_bio(tag, is_first)
            labels.append(label)
            previous = label
        examples.append(
            {
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "image": _render_image_from_tokens({"words": words, "bboxes": boxes}),
                "doc_id": str(record["id"]),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="Checkpoint folder.")
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tune", action="store_true", help="Sweep confidence thresholds.")
    args = parser.parse_args()

    model_path = args.model_path or "Drise Cord Fine-tuned Checkpoint"
    model, processor, device = load_model(model_path)
    print(f"Loaded model on {device}\n")

    examples = _build_examples(args.split)
    print(f"Evaluating {args.split} split: {len(examples)} documents\n")

    if args.tune:
        rows = sweep_confidence_thresholds(
            examples,
            model=model,
            processor=processor,
            device=device,
            thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
            batch_size=args.batch_size,
        )
        print(f"{'threshold':>10} | {'token_f1':>8} | {'entity_f1':>9}")
        print("-" * 34)
        for row in rows:
            entity = f"{row['entity_f1']:.4f}" if row["entity_f1"] is not None else "   n/a"
            print(f"{row['threshold']:>10.3f} | {row['token_f1']:>8.4f} | {entity:>9}")
        return 0

    metrics = evaluate_validation_set(
        examples,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )
    token = metrics["token_level"]
    entity = metrics["entity_level"]
    print(f"Documents            : {metrics['num_examples']}")
    token_row = (
        f"Token-level P/R/F1   : {token['precision']:.4f} / "
        f"{token['recall']:.4f} / {token['f1']:.4f}"
    )
    print(token_row)
    if entity.get("f1") is not None:
        entity_row = (
            f"Entity-level P/R/F1  : {entity['precision']:.4f} / "
            f"{entity['recall']:.4f} / {entity['f1']:.4f}"
        )
        print(entity_row)
    print(f"Mean non-O confidence: {metrics['mean_non_o_confidence']:.4f}")
    if entity.get("classification_report"):
        print("\n--- seqeval classification report ---\n")
        print(entity["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
