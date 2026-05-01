from __future__ import annotations

from document_intelligence_engine.pipelines.drise import DRISEPipeline
from document_intelligence_engine.pipelines.llm_only import LLMOnlyPipeline
from document_intelligence_engine.pipelines.rag_llm import RAGLLMPipeline


def test_llm_only_pipeline_uses_mock_backend(tmp_path):
    pipeline = LLMOnlyPipeline(
        {
            "backend": "mock",
            "model": "mock-llm",
            "cache_dir": str(tmp_path / "llm-cache"),
        }
    )

    output = pipeline.run(
        {
            "doc_id": "doc-1",
            "ocr_text": (
                "Invoice Number: INV-1023\nDate: 2025-01-12\nVendor: ABC Corp\n"
                "Item: Widget A, Qty: 2, Price: $500.00\nTotal Amount: $1000.00"
            ),
        }
    )

    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["extracted_fields"]["date"] == "2025-01-12"
    assert output["extracted_fields"]["vendor"] == "ABC Corp"
    assert output["extracted_fields"]["total_amount"] == 1000.0


def test_rag_pipeline_uses_retrieval_with_mock_backend(tmp_path):
    pipeline = RAGLLMPipeline(
        {
            "backend": "mock",
            "model": "mock-llm",
            "cache_dir": str(tmp_path / "llm-cache"),
            "retrieval_cache_dir": str(tmp_path / "retrieval-cache"),
            "top_k": 2,
        }
    )

    output = pipeline.run(
        {
            "doc_id": "doc-2",
            "ocr_text": (
                "Vendor: ABC Corp\nInvoice Number: INV-1023\nDate: 2025-01-12\n"
                "Item: Widget A, Qty: 2, Price: $500.00\nTotal Amount: $1000.00"
            ),
        }
    )

    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["extracted_fields"]["total_amount"] == 1000.0
    assert output["extracted_fields"]["line_items"] == [{"description": "Widget A", "quantity": 2.0, "unit_price": 500.0}]
    assert output["metadata"]["chunk_count"] >= 1
    assert output["metadata"]["top_k"] == 2
    assert "invoice_number" in output["metadata"]["retrieved_contexts"]


def test_rag_pipeline_repairs_embedded_json_field_output(tmp_path):
    class NoisyClient:
        def generate(self, prompt: str, **kwargs):
            if 'field "line_items"' in prompt:
                text = 'answer: [{"description":"Widget A","quantity":2.0,"unit_price":500.0}]'
            elif 'field "total_amount"' in prompt:
                text = "answer: 1000.0"
            elif 'field "invoice_number"' in prompt:
                text = 'answer: "INV-1023"'
            elif 'field "date"' in prompt:
                text = 'answer: "2025-01-12"'
            else:
                text = 'answer: "ABC Corp"'
            return {"text": text, "cost_usd": 0.0}

    pipeline = RAGLLMPipeline(
        {
            "backend": "mock",
            "model": "mock-llm",
            "retrieval_cache_dir": str(tmp_path / "retrieval-cache"),
        },
        client=NoisyClient(),  # type: ignore[arg-type]
    )

    output = pipeline.run(
        {
            "doc_id": "doc-embedded",
            "ocr_text": (
                "Vendor: ABC Corp\nInvoice Number: INV-1023\nDate: 2025-01-12\n"
                "Item: Widget A, Qty: 2, Price: $500.00\nTotal Amount: $1000.00"
            ),
        }
    )

    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["extracted_fields"]["total_amount"] == 1000.0
    assert output["extracted_fields"]["line_items"] == [{"description": "Widget A", "quantity": 2.0, "unit_price": 500.0}]


def test_drise_pipeline_flattens_existing_parser_output():
    class StubParserService:
        def parse_file(self, file_path, debug=False, use_layout=True, apply_constraints=True):
            return {
                "document": {
                    "invoice_number": {"value": "INV-1023", "confidence": 0.97, "valid": True},
                    "date": {"value": "2025-01-12", "confidence": 0.95, "valid": True},
                    "total_amount": {"value": 1200.5, "confidence": 0.93, "valid": True},
                    "_errors": [],
                    "_constraint_flags": ["ok"],
                },
                "metadata": {"timing": {"total": 8.5}},
            }

    pipeline = DRISEPipeline({"use_layout": False, "use_constraints": False}, parser_service=StubParserService())
    output = pipeline.run({"doc_id": "doc-3", "image_path": "/tmp/sample.png"})

    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["confidences"]["invoice_number"] == 0.97
    assert output["_constraint_flags"] == ["ok"]
    assert output["latency_ms"] == 8.5


def test_drise_pipeline_can_run_from_ocr_text_without_image_path():
    class StubModelService:
        name = "heuristic"
        version = "0.1.0"
        device = "cpu"

        def predict(self, ocr_tokens):
            return [
                {"text": "Invoice", "label": "B-KEY", "confidence": 0.99},
                {"text": "Number:", "label": "I-KEY", "confidence": 0.99},
                {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.99},
            ]

        def predict_text_only(self, ocr_tokens):
            return self.predict(ocr_tokens)

    class StubParserService:
        model_service = StubModelService()

        def parse_file(self, *args, **kwargs):
            raise AssertionError("parse_file should not be called when ocr_text is available")

    pipeline = DRISEPipeline({"use_layout": True, "use_constraints": True}, parser_service=StubParserService())
    output = pipeline.run({"doc_id": "doc-4", "ocr_text": "Invoice Number: INV-1023"})

    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["metadata"]["source"] == "experiment_ocr_tokens"
