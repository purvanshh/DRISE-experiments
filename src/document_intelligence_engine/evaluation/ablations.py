"""Ablation experiment definitions."""

from __future__ import annotations

from pydantic import BaseModel


class AblationRun(BaseModel):
    name: str
    description: str
    toggles: dict[str, bool]


def default_ablations() -> list[AblationRun]:
    return [
        AblationRun(
            name="without_preprocessing",
            description="Disable image normalization before OCR.",
            toggles={"preprocessing": False, "ocr": True, "layoutlmv3": True, "constraints": True},
        ),
        AblationRun(
            name="without_layoutlmv3",
            description="Use OCR-only baseline without multimodal classification.",
            toggles={"preprocessing": True, "ocr": True, "layoutlmv3": False, "constraints": True},
        ),
        AblationRun(
            name="without_constraints",
            description="Disable deterministic validation and constraint checks.",
            toggles={"preprocessing": True, "ocr": True, "layoutlmv3": True, "constraints": False},
        ),
    ]
