"""Chunking and retrieval utilities for RAG experiments."""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any

from document_intelligence_engine.retrieval.embedder import Embedder


class DocumentRetriever:
    """Cosine-similarity retriever with per-document cached indices."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        top_k: int = 3,
        chunk_size: int = 500,
        cache_dir: str | Path = "experiments/cache/retrieval",
    ) -> None:
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chunk_text(self, ocr_text: str, max_chars: int | None = None) -> list[str]:
        limit = max_chars or self.chunk_size
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for line in ocr_text.splitlines() or [ocr_text]:
            normalized = line.strip()
            if not normalized:
                continue
            projected_length = current_length + len(normalized) + (1 if current else 0)
            if current and projected_length > limit:
                chunks.append("\n".join(current))
                current = [normalized]
                current_length = len(normalized)
            else:
                current.append(normalized)
                current_length = projected_length
        if current:
            chunks.append("\n".join(current))
        fallback = ocr_text.strip()
        return chunks or ([fallback] if fallback else [])

    def build_index(self, doc_id: str, chunks: list[str]) -> Path:
        cache_path = self.cache_dir / f"{doc_id}.pkl"
        embeddings = self.embedder.encode(chunks)
        payload = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "chunk_hash": hashlib.sha256("\n\n".join(chunks).encode("utf-8")).hexdigest(),
            "chunks": chunks,
            "embeddings": embeddings,
        }
        with cache_path.open("wb") as file_pointer:
            pickle.dump(payload, file_pointer)
        return cache_path

    def load_index(self, doc_id: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{doc_id}.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"No retrieval index available for document '{doc_id}'.")
        with cache_path.open("rb") as file_pointer:
            payload = pickle.load(file_pointer)
        return dict(payload)

    def ensure_index(self, doc_id: str, ocr_text: str) -> list[str]:
        cache_path = self.cache_dir / f"{doc_id}.pkl"
        if cache_path.exists():
            payload = self.load_index(doc_id)
            return list(payload["chunks"])
        chunks = self.chunk_text(ocr_text)
        self.build_index(doc_id, chunks)
        return chunks

    def retrieve(self, query: str, doc_id: str, ocr_text: str | None = None) -> list[str]:
        try:
            payload = self.load_index(doc_id)
        except FileNotFoundError:
            if ocr_text is None:
                raise
            self.ensure_index(doc_id, ocr_text)
            payload = self.load_index(doc_id)

        chunks = list(payload["chunks"])
        embeddings = list(payload["embeddings"])
        query_embedding = self.embedder.encode([query])[0]
        ranked = sorted(
            ((self._cosine_similarity(query_embedding, embedding), index) for index, embedding in enumerate(embeddings)),
            reverse=True,
        )
        return [chunks[index] for _, index in ranked[: self.top_k]]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
