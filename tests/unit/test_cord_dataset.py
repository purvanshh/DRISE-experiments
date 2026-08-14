"""Unit tests for CORD dataset loader."""

from __future__ import annotations

import json

import pytest

from document_intelligence_engine.multimodal.cord_dataset import (
    LABEL2ID,
    LABEL_LIST,
    NUM_LABELS,
    _cord_label_to_bio,
    _normalize_bbox,
    _parse_cord_v2_example,
)


def test_cord_v2_parse_uses_valid_line():
    """cord-v2 stores geometry-bearing words in ``valid_line``; ``gt_parse``
    only holds field values without word boxes, so it must not be used for
    token supervision."""
    ground_truth = json.dumps({
        "gt_parse": {
            "menu": [{"nm": "Lemon Tea", "cnt": "1", "price": "7,000"}],
            "total": {"total_price": "7,000"},
        },
        "valid_line": [
            {
                "category": "menu.nm",
                "words": [
                    {
                        "text": "Lemon",
                        "quad": {"x1": 1, "y1": 2, "x2": 51, "y2": 2,
                                 "x3": 51, "y3": 22, "x4": 1, "y4": 22},
                        "is_key": False,
                    },
                    {
                        "text": "Tea",
                        "quad": {"x1": 51, "y1": 2, "x2": 81, "y2": 2,
                                 "x3": 81, "y3": 22, "x4": 51, "y4": 22},
                        "is_key": False,
                    },
                ],
            },
            {
                "category": "total.total_price",
                "words": [
                    {
                        "text": "Total",
                        "quad": {"x1": 1, "y1": 30, "x2": 41, "y2": 30,
                                 "x3": 41, "y3": 50, "x4": 1, "y4": 50},
                        "is_key": True,
                    },
                    {
                        "text": "7,000",
                        "quad": {"x1": 41, "y1": 30, "x2": 91, "y2": 30,
                                 "x3": 91, "y3": 50, "x4": 41, "y4": 50},
                        "is_key": False,
                    },
                ],
            },
        ],
    })
    parsed = _parse_cord_v2_example({"ground_truth": ground_truth})

    assert parsed["words"] == ["Lemon", "Tea", "Total", "7,000"]
    assert parsed["bio_labels"] == ["B-VALUE", "I-VALUE", "B-KEY", "I-VALUE"]
    assert parsed["boxes"] == [
        [1, 2, 51, 22],
        [51, 2, 81, 22],
        [1, 30, 41, 50],
        [41, 30, 91, 50],
    ]


def test_label_list_has_five_classes():
    assert NUM_LABELS == 5
    assert set(LABEL_LIST) == {"O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"}


def test_label2id_roundtrip():
    for label, idx in LABEL2ID.items():
        assert 0 <= idx < NUM_LABELS
        assert label in LABEL_LIST


def test_cord_label_to_bio_other():
    assert _cord_label_to_bio("O", is_first_token=True) == "O"
    assert _cord_label_to_bio("O", is_first_token=False) == "O"


def test_cord_label_to_bio_value():
    assert _cord_label_to_bio("menu.nm", is_first_token=True) == "B-VALUE"
    assert _cord_label_to_bio("menu.nm", is_first_token=False) == "I-VALUE"
    assert _cord_label_to_bio("total.total_price", is_first_token=True) == "B-VALUE"


def test_normalize_bbox_valid():
    result = _normalize_bbox([100, 200, 300, 400], width=1000, height=1000)
    assert result == [100, 200, 300, 400]


def test_normalize_bbox_scaling():
    result = _normalize_bbox([50, 100, 150, 200], width=500, height=500)
    assert result == [100, 200, 300, 400]


def test_normalize_bbox_zero_dimensions():
    result = _normalize_bbox([10, 20, 30, 40], width=0, height=0)
    assert result == [0, 0, 0, 0]


def test_normalize_bbox_clamped():
    result = _normalize_bbox([600, 700, 1200, 1400], width=1000, height=1000)
    assert result == [600, 700, 1000, 1000]
