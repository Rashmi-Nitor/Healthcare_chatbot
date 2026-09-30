"""
app/rag/vector_store.py

Phase 2 (part 3) — FAISS vector store creation & persistence.

Wraps LangChain's FAISS integration to:
- build a fresh index from chunked Documents,
- persist it to disk (vector_db/),
- load a persisted index back into memory,
- and add new documents to an existing index incrementally.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.constants import VECTOR_DB_DIR, FAISS_INDEX_NAME
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreNotFoundError(Exception):
    """Raised when trying to load a FAISS index that hasn't been built yet."""


def _index_path(persist_dir: Path = VECTOR_DB_DIR) -> Path:
    return Path(persist_dir)


def build_vector_store(
    chunks: List[Document],
    persist_dir: Path = VECTOR_DB_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """
    Build a brand-new FAISS index from `chunks` and persist it to disk.
    Overwrites any existing index at the same location.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    embedding_model = get_embedding_model()
    logger.info("Building FAISS index from %d chunk(s)...", len(chunks))
    store = FAISS.from_documents(chunks, embedding_model)

    path = _index_path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path), index_name=index_name)
    logger.info("FAISS index persisted to %s (index_name=%s).", path, index_name)
    return store


def load_vector_store(
    persist_dir: Path = VECTOR_DB_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    path = _index_path(persist_dir)
    faiss_file = path / f"{index_name}.faiss"
    if not path.exists() or not faiss_file.exists():
        raise VectorStoreNotFoundError(
            f"No FAISS index found at {path} (expected {index_name}.faiss). "
            "Run ingestion first."
        )

    embedding_model = get_embedding_model()
    store = FAISS.load_local(
        str(path),
        embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # safe: we only load files we wrote
    )
    logger.info("Loaded FAISS index from %s.", path)
    return store


def get_or_create_vector_store(
    chunks: Optional[List[Document]] = None,
    persist_dir: Path = VECTOR_DB_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """
    Convenience helper: load the index if it already exists, otherwise
    build it from `chunks` (which must be provided in that case).
    """
    try:
        return load_vector_store(persist_dir, index_name)
    except VectorStoreNotFoundError:
        if not chunks:
            raise
        logger.info("No existing index found — building a new one.")
        return build_vector_store(chunks, persist_dir, index_name)


def add_documents(
    store: FAISS,
    new_chunks: List[Document],
    persist_dir: Path = VECTOR_DB_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """Add new chunks to an existing FAISS store and re-persist it."""
    if not new_chunks:
        return store
    store.add_documents(new_chunks)
    path = _index_path(persist_dir)
    store.save_local(str(path), index_name=index_name)
    logger.info("Added %d new chunk(s) to the index and re-persisted.", len(new_chunks))
    return store


def reset_vector_store(persist_dir: Path = VECTOR_DB_DIR) -> None:
    """Delete a persisted index entirely (used by re-ingestion scripts/tests)."""
    path = _index_path(persist_dir)
    if path.exists():
        shutil.rmtree(path)
        logger.info("Deleted vector store at %s.", path)
