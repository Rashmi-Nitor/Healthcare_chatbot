"""
app/rag/retriever.py

Phase 3 — Semantic retrieval.

Provides top-k similarity search over the FAISS store, with a relevance
score threshold so we don't hand the LLM irrelevant chunks (which is how
hallucinated / off-topic answers sneak in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.constants import DEFAULT_TOP_K, SIMILARITY_SCORE_THRESHOLD, MAX_CONTEXT_CHARS

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document: Document
    score: float  # similarity score, higher = more relevant (0..1 range, cosine)

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self):
        return self.document.metadata.get("page")


class Retriever:
    """Thin, testable wrapper around a FAISS store's similarity search."""

    def __init__(self, store: FAISS, top_k: int = DEFAULT_TOP_K,
                 score_threshold: float = SIMILARITY_SCORE_THRESHOLD):
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """
        Run similarity search and return chunks above `score_threshold`,
        best first. Returns an empty list if nothing meets the bar — callers
        should treat that as "no grounded answer available".
        """
        k = top_k or self.top_k
        if not query or not query.strip():
            return []

        try:
            # FAISS returns L2 distance by default via similarity_search_with_score;
            # since we normalize embeddings, we convert to a cosine-similarity-like
            # score in [0, 1] for a consistent, interpretable threshold.
            raw_results = self.store.similarity_search_with_score(query, k=k)
        except Exception:
            logger.exception("Retrieval failed for query: %r", query)
            return []

        results: List[RetrievedChunk] = []
        for doc, distance in raw_results:
            similarity = max(0.0, 1.0 - (distance / 2.0))  # normalized L2 -> cosine approx
            if similarity >= self.score_threshold:
                results.append(RetrievedChunk(document=doc, score=similarity))
            else:
                logger.debug("Dropped low-relevance chunk (score=%.3f) from %s",
                             similarity, doc.metadata.get("source"))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("Retrieved %d/%d chunk(s) above threshold %.2f for query.",
                    len(results), len(raw_results), self.score_threshold)
        return results

    def build_context_string(self, chunks: List[RetrievedChunk],
                              max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Concatenate retrieved chunks into a single context string for the
        LLM prompt, each block tagged with its source so the model (and our
        citation post-processor) can attribute claims correctly.
        """
        blocks = []
        total = 0
        for chunk in chunks:
            block = f"[Source: {chunk.source} | page {chunk.page}]\n{chunk.document.page_content}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n---\n\n".join(blocks)


def build_retriever(store: FAISS, top_k: int = DEFAULT_TOP_K) -> Retriever:
    return Retriever(store, top_k=top_k)
