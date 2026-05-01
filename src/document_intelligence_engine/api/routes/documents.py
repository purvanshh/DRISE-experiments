"""Document processing routes."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from document_intelligence_engine.api.schemas.documents import DocumentParseResponse
from document_intelligence_engine.ingestion.validators import validate_upload
from document_intelligence_engine.services.pipeline import DocumentPipeline


router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("/parse", response_model=DocumentParseResponse)
async def parse_document(file: UploadFile = File(...)) -> DocumentParseResponse:
    validated = await validate_upload(file)
    pipeline = DocumentPipeline()
    result = pipeline.process(validated)
    return DocumentParseResponse.model_validate(result.model_dump())
