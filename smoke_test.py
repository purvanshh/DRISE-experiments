from __future__ import annotations

from transformers import AutoModelForTokenClassification, AutoProcessor


MODEL_NAME = "jinhybr/OCR-LayoutLMv3-Invoice"
EXPECTED_LABELS = {"O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"}


def remap_label(label: str) -> str:
    if label == "O":
        return "O"
    if label.startswith("B-"):
        return "B-VALUE"
    if label.startswith("I-"):
        return "I-VALUE"
    normalized = label.strip().lower()
    if normalized in {"ignore", "others"}:
        return "O"
    if normalized.endswith("_key") or normalized.endswith(".key"):
        return "B-KEY"
    if normalized.endswith("_value") or normalized.endswith(".value"):
        return "B-VALUE"
    return "O"


def main() -> None:
    print(f"Loading processor: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.eval()

    id2label = {int(label_id): label for label_id, label in model.config.id2label.items()}
    remapped_labels = sorted({remap_label(label) for label in id2label.values()})
    unexpected_labels = sorted(set(remapped_labels) - EXPECTED_LABELS)

    print(f"Processor class: {processor.__class__.__name__}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Native label count: {len(id2label)}")
    print("Native labels:")
    for label_id in sorted(id2label):
        print(f"  {label_id}: {id2label[label_id]}")

    print(f"Remapped label set: {remapped_labels}")
    print(f"Unknown label fallback: {remap_label('SOMETHING_UNEXPECTED')}")

    if unexpected_labels:
        raise SystemExit(f"Unexpected remapped labels: {unexpected_labels}")

    print("Load successful")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
