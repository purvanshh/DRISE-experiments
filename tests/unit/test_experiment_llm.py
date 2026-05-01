from __future__ import annotations

from document_intelligence_engine.llm.client import LLMClient
from document_intelligence_engine.pipelines.llm_only import LLMOnlyPipeline


def test_llm_client_caches_identical_prompt(tmp_path):
    client = LLMClient(backend="mock", model="mock-llm", cache_dir=tmp_path / "llm-cache")

    first = client.generate("OCR Text:\nInvoice Number: INV-1")
    second = client.generate("OCR Text:\nInvoice Number: INV-1")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_llm_client_extract_json_repairs_embedded_json(tmp_path):
    class BrokenClient(LLMClient):
        def generate(self, prompt: str, **kwargs):  # type: ignore[override]
            return {
                "text": 'Result: {"invoice_number":"INV-1","date":"2025-01-01","vendor":null,"total_amount":10.0,"line_items":[]}',
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "cost_usd": 0.0,
                "cache_hit": False,
            }

    client = BrokenClient(backend="mock", model="mock-llm", cache_dir=tmp_path / "llm-cache")
    payload = client.extract_json("ignored")

    assert payload["parsed"]["invoice_number"] == "INV-1"
    assert payload["parsed"]["total_amount"] == 10.0


def test_llm_only_pipeline_retries_after_failure(tmp_path):
    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def extract_json(self, prompt: str, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("temporary parse error")
            return {
                "text": '{"invoice_number":"INV-1023","date":"2025-01-12","vendor":"ABC Corp","total_amount":1000.0,"line_items":[]}',
                "cost_usd": 0.0,
                "parsed": {
                    "invoice_number": "INV-1023",
                    "date": "2025-01-12",
                    "vendor": "ABC Corp",
                    "total_amount": 1000.0,
                    "line_items": [],
                },
            }

    client = FlakyClient()
    pipeline = LLMOnlyPipeline(
        {
            "backend": "mock",
            "model": "mock-llm",
            "cache_dir": str(tmp_path / "llm-cache"),
            "max_retries": 2,
        },
        client=client,  # type: ignore[arg-type]
    )

    output = pipeline.run({"doc_id": "retry-doc", "ocr_text": "Invoice Number: INV-1023"})

    assert client.calls == 2
    assert output["extracted_fields"]["invoice_number"] == "INV-1023"
    assert output["_errors"] == ["attempt_1: temporary parse error"]
