"""Unit tests for the FAISS vector store and the retrieve() pipeline (fully offline)."""
from __future__ import annotations

import numpy as np

from customer_intelligence.rag.vector_store import VectorStore
from customer_intelligence.rag.ingest import Chunk
from customer_intelligence.config import settings


def test_vector_store_add_and_search_returns_closest_match():
    store = VectorStore(dim=4)
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype="float32")
    store.add(vectors, ["a", "b", "c"])

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    results = store.search(query, k=2)

    assert results[0][0] == "a"
    assert 0.0 <= results[0][1] <= 1.0
    assert len(results) == 2


def test_vector_store_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dim=3)
    store.add(np.array([[1.0, 0.0, 0.0]], dtype="float32"), ["only"])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.size == 1
    results = loaded.search(np.array([1.0, 0.0, 0.0], dtype="float32"), k=1)
    assert results[0][0] == "only"


def test_retrieve_returns_citations_from_a_prebuilt_index(tmp_path, monkeypatch, fake_bedrock):
    from customer_intelligence.rag import retriever

    # Build a tiny prebuilt index + metadata sidecar, matching what `ingest.build_index` produces.
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    store = VectorStore(dim=16)
    embeddings = np.array(fake_bedrock.embed(["billing refund policy chunk"]), dtype="float32")
    store.add(embeddings, ["refunds#0"])
    store.save(rag_dir)

    import json
    from dataclasses import asdict
    chunk = Chunk(chunk_id="refunds#0", title="Billing Policy", source="kb", text="Refunds are processed within 5 days.")
    with open(rag_dir / "kb_meta.json", "w", encoding="utf-8") as f:
        json.dump({"refunds#0": asdict(chunk)}, f)

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    retriever.reload_index()

    citations = retriever.retrieve("What is the refund policy?", k=1)

    assert len(citations) == 1
    assert citations[0].chunk_id == "refunds#0"
    assert citations[0].title == "Billing Policy"

    retriever.reload_index()
