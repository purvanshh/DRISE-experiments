"""Embedding wrapper with an offline hashing fallback."""

from __future__ import annotations

import hashlib
import math
import re


class Embedder:
    """Sentence embedding wrapper used by the RAG baseline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_size: int = 256) -> None:
        self.model_name = model_name
        self.vector_size = vector_size
        self._model = None
        if model_name != "hashing":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.model_name = "hashing"
            else:  # pragma: no cover - depends on optional dependency
                try:
                    self._model = SentenceTransformer(model_name)
                except Exception:
                    self.model_name = "hashing"
                    self._model = None
        if self.model_name == "hashing":
            self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:  # pragma: no cover - depends on optional dependency
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        return [self._hashing_encode(text) for text in texts]

    def _hashing_encode(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.vector_size
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
