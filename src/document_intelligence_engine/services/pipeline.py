"""End-to-end document processing pipeline."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.contracts import DocumentProcessingResult, ValidatedFile
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


class DocumentPipeline:
    def __init__(self) -> None:
        settings = get_settings()
        model_service = LayoutAwareModelService(settings)
        model_service.load()
        self.parser_service = DocumentParserService(settings, model_service)

    def process(self, document: ValidatedFile) -> DocumentProcessingResult:
        suffix = Path(document.safe_name).suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="document-pipeline-") as temporary_directory:
            temp_path = Path(temporary_directory) / f"{document.sha256}{suffix}"
            temp_path.write_bytes(document.payload)
            result = self.parser_service.parse_file(temp_path)

        return DocumentProcessingResult(
            document_id=document.sha256,
            status="processed",
            source_name=document.safe_name,
            content_type=document.content_type,
            page_count=int(result["metadata"]["page_count"]),
            extracted=result["document"],
            constraint_flags=list(result["document"].get("_constraint_flags", [])),
            ocr_engine=get_settings().ocr.backend,
            model_name=self.parser_service.model_service.name,
            processed_at=datetime.now(timezone.utc),
        )
