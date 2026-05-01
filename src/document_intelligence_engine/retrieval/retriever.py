"""Chunking and retrieval utilities for RAG experiments."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

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
        return chunks or [ocr_text.strip()]

    def ensure_index(self, doc_id: str, ocr_text: str) -> list[str]:
        cache_path = self.cache_dir / f"{doc_id}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as file_pointer:
                payload = pickle.load(file_pointer)
            return list(payload["chunks"])

        chunks = self.chunk_text(ocr_text)
        embeddings = self.embedder.encode(chunks)
        with cache_path.open("wb") as file_pointer:
            pickle.dump({"chunks": chunks, "embeddings": embeddings}, file_pointer)
        return chunks

    def retrieve(self, query: str, doc_id: str, ocr_text: str | None = None) -> list[str]:
        cache_path = self.cache_dir / f"{doc_id}.pkl"
        if not cache_path.exists():
            if ocr_text is None:
                raise FileNotFoundError(f"No retrieval index available for document '{doc_id}'.")
            self.ensure_index(doc_id, ocr_text)

        with cache_path.open("rb") as file_pointer:
            payload = pickle.load(file_pointer)

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
