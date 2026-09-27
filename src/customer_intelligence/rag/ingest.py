"""
RAG knowledge base ingestion:
load_docs   - read Markdown files with YAML front-matter (title, source)
chunk       - recursive character splitter
build_index - embed chunks with the embedding model, persist FAISS index + metadata JSON
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import frontmatter

from ..bedrock import bedrock
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

#__________ Data Classes _______________________________________________________________

@dataclass
class Doc:
    title: str
    source: str
    text: str
    path: str = ""

@dataclass
class Chunk:
    chunk_id: str
    title: str
    source: str
    text: str

#_____________Document Loading___________________________________________________________
def load_docs(kb_dir: str | Path) -> list[Doc]:
    """
    Load all .md files from kb_dir (recurse into subdirectories).
    Each file should have YAML front-matter with `title` and `source`.
    """
    kb_dir = Path(kb_dir)
    docs: list[Doc] = []

    for md_path in sorted(kb_dir.rglob("*.md")):
        try:
            post = frontmatter.load(str(md_path))
            title = str(post.metadata.get("title", md_path.stem.replace("_", " ").title()))
            source = str(post.metadata.get("source", "internal"))
            text = post.content.strip()
            if text:
                docs.append(Doc(title=title, source=source, text=text, path=str(md_path)))
                logger.debug("Loaded doc %s (%d chars)", title, len(text))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", md_path, exc)
    logger.info("Loaded %d documents from %s", len(docs), kb_dir)
    return docs

#________Chunking________________________________________________________________________
# Prefer splitting on markdown section headers first, so each FAQ/policy subsection stays intact.
_SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " ", ""]
def _split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """
    Recursive character splitter - tries to split on paragraph/sentence boundaries.
    """
    if len(text) <= size:
        return [text]
    # find the best separator that splits within bounds
    for sep in _SEPARATORS:
        if sep and sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if overlap and current:
                        overlap_text = current[-overlap:]
                        current = overlap_text + (sep if overlap_text else "") + part
                    else:
                        current = part
            if current:
                chunks.append(current)
            if chunks:
                return chunks

    return [text[i: i + size] for i in range(0, len(text), size - overlap)]

def chunk(doc: Doc, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split a doc into chunks with IDs."""
    slug = re.sub(r"[^a-z0-9]+", "_", doc.title.lower()).strip("_")
    texts = _split_text(doc.text, size=size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{slug}#{i}",
            title=doc.title,
            source=doc.source,
            text=t.strip(),
        )
        for i, t in enumerate(texts)
        if t.strip()
    ]

#__________Index Building__________________________________________________________________
def build_index(
        kb_dir: str | Path,
        out_dir: str | Path,
        chunk_size: int = 400,
        overlap: int = 60,
) -> tuple[VectorStore, list[Chunk]]:
    """
    Load docs -> chunk -> embed -> build FAISS index.
    Persists: faiss.index, faiss_ids.json, kb_meta.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(kb_dir)
    if not docs:
        raise ValueError(f"No markdown documents found in {kb_dir}")
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk(doc, size=chunk_size, overlap=overlap))

    logger.info("Created %d chunks from %d docs", len(all_chunks), len(docs))

    # Embed in batches of 20
    texts = [c.text for c in all_chunks]
    all_embeddings: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info("Embedding chunks %d-%d / %d...", i + 1, min(i + batch_size, len(texts)), len(texts))
        embs = bedrock.embed(batch)
        all_embeddings.extend(embs)

    embeddings = np.array(all_embeddings, dtype="float32")
    store = VectorStore()
    store.add(embeddings, [c.chunk_id for c in all_chunks])
    store.save(out_dir)
    # save metadata sidebar
    meta = {c.chunk_id: asdict(c) for c in all_chunks}
    with open(out_dir / "kb_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Index built: %d vectors -> %s", store.size, out_dir)
    return store, all_chunks

def load_index(out_dir: str | Path) -> tuple[VectorStore, dict[str, Chunk]]:
    """Load prebuilt index and metadata."""
    out_dir = Path(out_dir)
    store = VectorStore.load(out_dir)

    with open(out_dir / "kb_meta.json", encoding="utf-8") as f:
        raw_meta: dict = json.load(f)
    meta = {k: Chunk(**v) for k, v in raw_meta.items()}
    return store, meta
