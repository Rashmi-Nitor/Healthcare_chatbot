"""
app/rag/embeddings.py

Phase 2 (part 2) — Embedding generation.

Provides a single, cached factory for the sentence-transformers embedding
model (all-MiniLM-L6-v2) so the model is loaded into memory only once per
process, regardless of how many modules import it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.constants import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a process-wide singleton HuggingFaceEmbeddings instance.
    Cached with lru_cache so the (relatively expensive) model load only
    happens once.
    """
    logger.info("Loading embedding model '%s' on device '%s'...",
                EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},  # cosine-similarity friendly
    )
    logger.info("Embedding model ready.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience helper: embed a batch of raw strings directly."""
    if not texts:
        return []
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string (uses the query-side encode path)."""
    model = get_embedding_model()
    return model.embed_query(text)
