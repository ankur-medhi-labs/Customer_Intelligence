"""
FAISS-backed vector store with a JSON metadata sidebar.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class VectorStore:
    """
    Thin FAISS wrapper.

    add(vectors, ids)   - bulk-add normalised float32 vectors
    search(query, k)    - return top-k (chunk_id, cosine_score) pairs
    save(dir)           - persist index + metadata JSON
    load(dir)           - restore from disk
    """

    def __init__(self, dim: int | None = None) -> None:
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is required. run: pip install faiss-cpu")
        self.dim = dim or settings.embedding_dimensions
        self._index = faiss.IndexFlatIP(self.dim)
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add L2 normalised float32 vectors with corresponding chunk ids."""
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        normed = self._normalise(vectors)
        self._index.add(normed)
        self._ids.extend(ids)
        logger.info("Added %d vectors; total=%d", len(ids), len(self._ids))

    def search(self, query: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
        """
        Return top-k (chunk_id, cosine_score) pairs.
        Scores are in [0, 1] - higher is more similar.
        """
        if self._index.ntotal == 0:
            return []
        q = self._normalise(query.reshape(1, -1))
        k = min(k, self._index.ntotal)
        scores, positions = self._index.search(q, k)
        results: list[tuple[str, float]] = []
        for score, pos in zip(scores[0], positions[0]):
            if 0 <= pos < len(self._ids):
                results.append((self._ids[pos], float(min(1.0, max(0.0, score)))))
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "faiss.index"))
        with open(directory / "faiss_ids.json", "w", encoding="utf-8") as f:
            json.dump(self._ids, f)

        logger.info("Saved FAISS index (%d vectors) -> %s", len(self._ids), directory)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        store = cls()
        store._index = faiss.read_index(str(directory / "faiss.index"))
        with open(directory / "faiss_ids.json", encoding="utf-8") as f:
            store._ids = json.load(f)
        logger.info("Loaded FAISS index (%d vectors) from %s", len(store._ids), directory)
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype("float32")
        norms = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-10
        return arr / norms