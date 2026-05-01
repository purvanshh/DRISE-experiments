"""Unit tests for CORD dataset loader."""

from __future__ import annotations

import pytest

from document_intelligence_engine.multimodal.cord_dataset import (
    LABEL2ID,
    LABEL_LIST,
    NUM_LABELS,
    _cord_label_to_bio,
    _normalize_bbox,
)


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
