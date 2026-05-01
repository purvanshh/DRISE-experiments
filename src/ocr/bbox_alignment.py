"""Bounding box normalization and token alignment."""

from __future__ import annotations

from typing import Any

from ingestion.exceptions import BoundingBoxAlignmentError


def normalize_bbox(bbox: list[int] | tuple[int, int, int, int], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise BoundingBoxAlignmentError("Image dimensions must be positive.")
    x1, y1, x2, y2 = bbox
    x1 = _scale_coordinate(x1, width)
    y1 = _scale_coordinate(y1, height)
    x2 = _scale_coordinate(x2, width)
    y2 = _scale_coordinate(y2, height)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def align_tokens_with_boxes(tokens: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    if not tokens:
        return []

    width, height = image_size
    aligned: list[dict[str, Any]] = []
    for token in tokens:
        text = str(token.get("text", "")).strip()
        bbox = token.get("bbox")
        confidence = float(token.get("confidence", 0.0))
        if not text or not bbox or len(bbox) != 4:
            continue

        normalized_bbox = normalize_bbox([int(value) for value in bbox], width, height)
        aligned.append(
            {
                "text": text,
                "bbox": normalized_bbox,
                "confidence": round(confidence, 6),
            }
        )

    aligned.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["confidence"]))
    return _deduplicate_overlaps(aligned)


def _deduplicate_overlaps(tokens: list[dict[str, Any]], iou_threshold: float = 0.85) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    for token in tokens:
        if any(_iou(token["bbox"], existing["bbox"]) >= iou_threshold for existing in deduplicated):
            continue
        deduplicated.append(token)
    return deduplicated


def _iou(box_a: list[int], box_b: list[int]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    intersection_width = max(0, min(xa2, xb2) - max(xa1, xb1))
    intersection_height = max(0, min(ya2, yb2) - max(ya1, yb1))
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0
    box_a_area = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    box_b_area = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = box_a_area + box_b_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


def _scale_coordinate(value: int, dimension: int) -> int:
    return max(0, min(1000, int(round((value / dimension) * 1000))))
