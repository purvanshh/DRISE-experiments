"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool = Field(examples=[True])
    ocr_loaded: bool = Field(examples=[True])
    version: str = Field(examples=["0.1.0"])


class ErrorResponse(BaseModel):
    error: str = Field(examples=["Invalid file format"])
    code: int = Field(examples=[400])
    request_id: str | None = Field(default=None, examples=["req-123456"])
    details: list[dict[str, Any]] = Field(default_factory=list)


class ConfidenceSummary(BaseModel):
    average: float = Field(examples=[0.91])
    minimum: float = Field(examples=[0.84])
    maximum: float = Field(examples=[0.97])
    kept_fields: int = Field(examples=[3])
    dropped_fields: int = Field(examples=[1])


class ResponseMetadata(BaseModel):
    request_id: str = Field(examples=["req-123456"])
    filename: str = Field(examples=["invoice.pdf"])
    content_type: str = Field(examples=["application/pdf"])
    size_bytes: int = Field(examples=[24576])
    processing_time_ms: float = Field(examples=[128.44])
    timing: dict[str, float] = Field(
        examples=[{"validation": 1.2, "load": 25.4, "preprocessing": 12.0, "ocr": 140.1, "model": 4.1, "postprocessing": 1.3, "total": 184.1}]
    )
    confidence_summary: ConfidenceSummary
    page_count: int = Field(examples=[2])
    warnings: list[str] = Field(default_factory=list, examples=[["noisy_ocr_output"]])
    model: dict[str, str] = Field(
        examples=[{"name": "microsoft/layoutlmv3-base", "version": "0.1.0", "device": "cpu"}]
    )


class ParseDocumentResponse(BaseModel):
    document: dict[str, Any]
    metadata: ResponseMetadata

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": {
                    "invoice_number": {
                        "value": "INV-1023",
                        "confidence": 0.93,
                        "valid": True,
                    },
                    "total_amount": {
                        "value": 1200.5,
                        "confidence": 0.88,
                        "valid": True,
                    },
                    "_errors": [],
                    "_constraint_flags": [],
                },
                "metadata": {
                    "request_id": "req-123456",
                    "filename": "invoice.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 24576,
                    "processing_time_ms": 128.44,
                    "timing": {
                        "validation": 1.2,
                        "load": 25.4,
                        "preprocessing": 12.0,
                        "ocr": 140.1,
                        "model": 4.1,
                        "postprocessing": 1.3,
                        "total": 184.1
                    },
                    "confidence_summary": {
                        "average": 0.905,
                        "minimum": 0.88,
                        "maximum": 0.93,
                        "kept_fields": 2,
                        "dropped_fields": 0,
                    },
                    "page_count": 1,
                    "warnings": [],
                    "model": {
                        "name": "microsoft/layoutlmv3-base",
                        "version": "0.1.0",
                        "device": "cpu"
                    }
                },
            }
        }
    )


class BatchParseResponse(BaseModel):
    items: list[ParseDocumentResponse]


class ParseRequestMeta(BaseModel):
    request_id: str
