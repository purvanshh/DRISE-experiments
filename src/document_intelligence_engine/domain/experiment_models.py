"""Domain models shared by experiment pipelines and evaluation."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class ProcessedOCRToken(TypedDict, total=False):
    text: str
    bbox: list[int] | dict[str, int]
    confidence: float
    page_number: int


class ProcessedOCRMetadata(TypedDict, total=False):
    source: str
    engine: str
    language: str
    page_count: int
    token_count: int
    timing: dict[str, float]
    reused_cached_ocr: bool


class ProcessedDocument(TypedDict, total=False):
    doc_id: str
    image_path: str
    raw_image_path: str
    ocr_text: str
    ocr_tokens: list[ProcessedOCRToken]
    ocr_metadata: ProcessedOCRMetadata
    ground_truth: dict[str, Any] | None


class ExtractionOutput(TypedDict, total=False):
    extracted_fields: dict[str, Any]
    confidences: dict[str, float]
    _constraint_flags: list[str]
    _errors: list[str]
    latency_ms: float
    cost_usd: float
    raw_response: str
    metadata: dict[str, Any]


@runtime_checkable
class PipelineBase(Protocol):
    """Protocol implemented by experiment pipelines."""

    name: str

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        """Extract structured fields from a processed document."""
        ...
