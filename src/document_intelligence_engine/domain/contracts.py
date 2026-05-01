"""Typed contracts shared across modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)


class OCRToken(BaseModel):
    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    page_number: int = Field(ge=1)


class OCRResult(BaseModel):
    tokens: list[OCRToken]
    engine: str
    language: str


class ModelPrediction(BaseModel):
    labels: list[str]
    confidences: list[float]
    entities: dict[str, str | float | int | None]
    model_name: str


class ConstraintResult(BaseModel):
    normalized: dict[str, object]
    flags: list[str]


class ValidatedFile(BaseModel):
    original_name: str
    safe_name: str
    content_type: str
    extension: str
    size_bytes: int = Field(gt=0)
    sha256: str
    payload: bytes


class StoredDocument(BaseModel):
    path: Path
    metadata: ValidatedFile


class IngestedPage(BaseModel):
    page_number: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    image_bytes: bytes


class DocumentProcessingResult(BaseModel):
    document_id: str
    status: Literal["processed", "failed"]
    source_name: str
    content_type: str
    page_count: int = Field(ge=0)
    extracted: dict[str, object]
    constraint_flags: list[str]
    ocr_engine: str
    model_name: str
    processed_at: datetime
