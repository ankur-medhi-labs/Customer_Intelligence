"""
RAG retriever - top-k semantic search with a relevance gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bedrock import bedrock
from ..config import settings
from ..schemas import Citation
from .ingest import Chunk, load_index
from .vector_store import VectorStore
from ..logging_setup import get_logger

logger = get_logger(__name__)

# Module-level cached store and metadata
_store: VectorStore | None = None
_meta: dict[str, Chunk] = {}

def _get_store() -> tuple[VectorStore, dict[str, Chunk]]:
    global _store, _meta
    if _store is None:
        out_dir = Path(settings.artifacts_dir) / "rag"
        if not out_dir.exists():
            raise RuntimeError(
                f"RAG index not found at {out_dir}. "
                "Run `ci ingest` first to build the knowledge base index."
            )
        _store, _meta = load_index(out_dir)
    return _store, _meta

def retrieve(question: str, k: int | None = None) -> list[Citation]:
    """
    Embed the question and retrieve the top-k matching chunks.
    """
    k = k or settings.retrieval_top_k
    store, meta = _get_store()
    if store.size == 0:
        logger.warning("[retrieve] vector store is empty - no results")
        return []
    logger.debug("[retrieve] question=%r k=%d store_size=%d", question[:80], k, store.size)

    q_emb = np.array(bedrock.embed([question])[0], dtype="float32")
    hits = store.search(q_emb, k=k)
    citations: list[Citation] = []

    for chunk_id, score in hits:
        chunk = meta.get(chunk_id)
        if chunk is None:
            logger.warning("[retrieve] chunk_id %r not found in metadata", chunk_id)
            continue
        snippet = chunk.text[:300].replace("\n", " ").strip()
        citations.append(Citation(
            source=chunk.source,
            title=chunk.title,
            chunk_id=chunk_id,
            score=round(score, 4),
            snippet=snippet,
        ))

    return citations

def reload_index() -> None:
    """Force reload of the cached index."""
    global _store, _meta
    _store = None
    _meta = {}

