"""Experiment wrapper for the existing deterministic DRISE pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.experiment_models import ExtractionOutput, ProcessedDocument
from document_intelligence_engine.pipelines.base import BasePipeline
from document_intelligence_engine.services.document_parser import build_confidence_summary, derive_warnings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from postprocessing.pipeline import postprocess_predictions


REQUIRED_FIELDS = ("invoice_number", "date", "vendor", "total_amount", "line_items")


class DRISEPipeline(BasePipeline):
    """Wrap the existing parser service behind the experiment interface."""

    name = "drise"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        parser_service: DocumentParserService | None = None,
    ) -> None:
        self.config = config or {}
        self.name = str(self.config.get("name", self.name))
        self._settings = get_settings()
        self._parser_service = parser_service or self._build_parser_service()
        self._local_cost_per_hour = float(self.config.get("local_cost_per_hour", 0.0))

    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        ocr_tokens = list(document.get("ocr_tokens", []))
        if ocr_tokens:
            return self._run_from_ocr_tokens(document, ocr_tokens)

        ocr_text = str(document.get("ocr_text", "")).strip()
        if ocr_text:
            synthetic_tokens = _tokens_from_ocr_text(ocr_text)
            return self._run_from_ocr_tokens(document, synthetic_tokens)

        image_path = document.get("image_path")
        if not image_path:
            raise ValueError("DRISEPipeline requires document['image_path'], 'ocr_tokens', or 'ocr_text'.")

        started_at = time.perf_counter()
        result = self._parser_service.parse_file(
            image_path,
            debug=bool(self.config.get("debug", False)),
            use_layout=bool(self.config.get("use_layout", True)),
            apply_constraints=bool(self.config.get("use_constraints", True)),
        )
        latency_ms = result["metadata"]["timing"].get("total") or round((time.perf_counter() - started_at) * 1000, 3)
        extracted_fields, confidences = _flatten_document(result["document"])
        extracted_fields, confidences = _ensure_required_output_fields(extracted_fields, confidences)
        cost_usd = (float(latency_ms) / 1000.0 / 3600.0) * self._local_cost_per_hour

        return {
            "extracted_fields": extracted_fields,
            "confidences": confidences,
            "_constraint_flags": list(result["document"].get("_constraint_flags", [])),
            "_errors": [str(error) for error in result["document"].get("_errors", [])],
            "latency_ms": float(latency_ms),
            "cost_usd": round(cost_usd, 6),
            "metadata": result["metadata"],
        }

    def _run_from_ocr_tokens(
        self,
        document: ProcessedDocument,
        ocr_tokens: list[dict[str, Any]],
    ) -> ExtractionOutput:
        started_at = time.perf_counter()
        use_layout = bool(self.config.get("use_layout", True))
        apply_constraints = bool(self.config.get("use_constraints", True))
        repair_constraints = bool(self.config.get("repair_constraints", apply_constraints))
        model_service = self._parser_service.model_service
        page_image = _load_page_image(document.get("image_path"))
        ocr_metadata = dict(document.get("ocr_metadata", {}))

        model_started_at = time.perf_counter()
        prediction_method = model_service.predict if use_layout else model_service.predict_text_only
        raw_predictions = _invoke_prediction_method(
            prediction_method,
            ocr_tokens=ocr_tokens,
            page_image=page_image,
            use_layout=use_layout,
        )
        model_duration_ms = round((time.perf_counter() - model_started_at) * 1000, 3)

        postprocessing_started_at = time.perf_counter()
        structured_document = postprocess_predictions(
            raw_predictions,
            apply_constraints=apply_constraints,
            repair_constraints=repair_constraints,
            ocr_tokens=ocr_tokens,
        )
        postprocessing_duration_ms = round((time.perf_counter() - postprocessing_started_at) * 1000, 3)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)

        extracted_fields, confidences = _flatten_document(structured_document)
        extracted_fields, confidences = _ensure_required_output_fields(extracted_fields, confidences)
        cost_usd = (float(latency_ms) / 1000.0 / 3600.0) * self._local_cost_per_hour
        metadata = {
            "filename": document.get("image_path") or document.get("doc_id", "document"),
            "page_count": int(ocr_metadata.get("page_count", 1) or 1),
            "ocr_token_count": len(ocr_tokens),
            "confidence_summary": build_confidence_summary(structured_document),
            "timing": {
                "validation": float(ocr_metadata.get("timing", {}).get("validation", 0.0)),
                "load": float(ocr_metadata.get("timing", {}).get("load", 0.0)),
                "preprocessing": float(ocr_metadata.get("timing", {}).get("preprocessing", 0.0)),
                "ocr": float(ocr_metadata.get("timing", {}).get("ocr", 0.0)),
                "bbox_alignment": float(ocr_metadata.get("timing", {}).get("bbox_alignment", 0.0)),
                "model": model_duration_ms,
                "postprocessing": postprocessing_duration_ms,
                "total": latency_ms,
            },
            "warnings": derive_warnings(
                ocr_tokens=ocr_tokens,
                document=structured_document,
                page_count=int(ocr_metadata.get("page_count", 1) or 1),
            ),
            "model": {
                "name": model_service.name,
                "version": model_service.version,
                "device": model_service.device,
            },
            "source": str(ocr_metadata.get("source", "experiment_ocr_tokens")),
            "ocr": ocr_metadata,
        }

        return {
            "extracted_fields": extracted_fields,
            "confidences": confidences,
            "_constraint_flags": list(structured_document.get("_constraint_flags", [])),
            "_errors": [str(error) for error in structured_document.get("_errors", [])],
            "latency_ms": float(latency_ms),
            "cost_usd": round(cost_usd, 6),
            "metadata": metadata,
        }

    def _build_parser_service(self) -> DocumentParserService:
        settings = self._settings
        model_service = LayoutAwareModelService(settings)
        model_service.load()
        return DocumentParserService(settings, model_service)


def _flatten_document(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    extracted_fields: dict[str, Any] = {}
    confidences: dict[str, float] = {}
    for field_name, payload in document.items():
        if field_name.startswith("_"):
            continue
        if isinstance(payload, dict) and "value" in payload:
            extracted_fields[field_name] = payload.get("value")
            confidences[field_name] = round(float(payload.get("confidence", 0.0)), 6)
        else:
            extracted_fields[field_name] = payload
            confidences[field_name] = 0.0
    return extracted_fields, confidences


def _tokens_from_ocr_text(ocr_text: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    y_offset = 0
    for line in ocr_text.splitlines():
        x_offset = 0
        for raw_token in line.split():
            width = max(12, len(raw_token) * 8)
            tokens.append(
                {
                    "text": raw_token,
                    "bbox": [x_offset, y_offset, x_offset + width, y_offset + 16],
                    "confidence": 0.99,
                    "page_number": 1,
                }
            )
            x_offset += width + 6
        y_offset += 24
    return tokens


def _ensure_required_output_fields(
    extracted_fields: dict[str, Any],
    confidences: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    normalized_fields = dict(extracted_fields)
    normalized_confidences = dict(confidences)

    for field_name in REQUIRED_FIELDS:
        if field_name not in normalized_fields:
            normalized_fields[field_name] = [] if field_name == "line_items" else None
        if field_name == "line_items" and normalized_fields[field_name] is None:
            normalized_fields[field_name] = []
        normalized_confidences.setdefault(field_name, 0.0)

    return normalized_fields, normalized_confidences


def _invoke_prediction_method(
    prediction_method: Any,
    *,
    ocr_tokens: list[dict[str, Any]],
    page_image: Image.Image | None,
    use_layout: bool,
) -> Any:
    if not use_layout:
        return prediction_method(ocr_tokens)
    try:
        return prediction_method(ocr_tokens, page_image=page_image)
    except TypeError as exc:
        if "page_image" not in str(exc):
            raise
        return prediction_method(ocr_tokens)


def _load_page_image(image_path: Any) -> Image.Image | None:
    if not image_path:
        return None
    path = Path(str(image_path)).expanduser()
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")
