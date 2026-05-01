from __future__ import annotations

from document_intelligence_engine.retrieval.embedder import Embedder
from document_intelligence_engine.retrieval.retriever import DocumentRetriever


def test_retriever_chunks_text_by_line_groups(tmp_path):
    retriever = DocumentRetriever(
        embedder=Embedder("hashing"),
        chunk_size=25,
        cache_dir=tmp_path / "retrieval-cache",
    )

    chunks = retriever.chunk_text("Invoice Number: INV-1\nDate: 2025-01-01\nTotal Amount: 25.00")

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_retriever_builds_and_loads_cached_index(tmp_path):
    retriever = DocumentRetriever(
        embedder=Embedder("hashing"),
        cache_dir=tmp_path / "retrieval-cache",
    )

    chunks = retriever.chunk_text("Invoice Number: INV-1\nDate: 2025-01-01")
    cache_path = retriever.build_index("doc-1", chunks)
    payload = retriever.load_index("doc-1")

    assert cache_path.exists()
    assert payload["doc_id"] == "doc-1"
    assert payload["chunk_count"] == len(chunks)
    assert payload["chunks"] == chunks


def test_retriever_returns_relevant_chunks(tmp_path):
    retriever = DocumentRetriever(
        embedder=Embedder("hashing"),
        top_k=1,
        cache_dir=tmp_path / "retrieval-cache",
    )

    retriever.ensure_index(
        "doc-2",
        "Invoice Number: INV-100\nVendor: ABC Corp\nDate: 2025-01-01\nTotal Amount: 50.00",
    )
    results = retriever.retrieve("vendor merchant name", "doc-2")

    assert len(results) == 1
    assert "Vendor: ABC Corp" in results[0]
