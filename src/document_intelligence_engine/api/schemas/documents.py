"""Document API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentParseResponse(BaseModel):
    document_id: str
    status: str
    source_name: str
    content_type: str
    page_count: int
    extracted: dict[str, object]
    constraint_flags: list[str]
    ocr_engine: str
    model_name: str
    processed_at: datetime
