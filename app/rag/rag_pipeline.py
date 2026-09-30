"""
app/rag/rag_pipeline.py

Ties data_loader -> text_splitter -> embeddings -> vector_store -> retriever
together into two simple entry points:

    ingest_knowledge_base()  -> builds/refreshes the FAISS index from data/
    RAGPipeline.retrieve()   -> runs retrieval + builds context + citations
                                for a single user query (used by
                                llm/healthcare_assistant.py)

Keeping this orchestration separate from `healthcare_assistant.py` means the
retrieval half of RAG can be tested/used independently of Gemini/guardrails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from app.core.constants import DATA_DIR, VECTOR_DB_DIR, FAISS_INDEX_NAME, DEFAULT_TOP_K
from app.rag.data_loader import load_directory
from app.rag.text_splitter import split_documents
from app.rag.vector_store import build_vector_store, get_or_create_vector_store, VectorStoreNotFoundError
from app.rag.retriever import build_retriever, RetrievedChunk
from app.rag.citation_generator import generate_citations, Citation

logger = logging.getLogger(__name__)


def ingest_knowledge_base(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTOR_DB_DIR,
    index_name: str = FAISS_INDEX_NAME,
) -> int:
    """
    Full ingestion run: load every supported file in `data_dir`, chunk it,
    embed it, and (re)build the persisted FAISS index.

    Returns the number of chunks indexed.
    """
    logger.info("Starting knowledge-base ingestion from %s", data_dir)
    documents = load_directory(data_dir)
    if not documents:
        logger.warning("No documents loaded — ingestion aborted.")
        return 0

    chunks = split_documents(documents)
    build_vector_store(chunks, persist_dir=persist_dir, index_name=index_name)
    logger.info("Ingestion complete: %d chunks indexed.", len(chunks))
    return len(chunks)


class RAGPipeline:
    """
    Query-time half of RAG: given a question, retrieve relevant chunks,
    build an LLM-ready context string, and generate citations.

    Loads the FAISS index lazily on first use so importing this module
    doesn't require the index to already exist.
    """

    def __init__(self, persist_dir: Path = VECTOR_DB_DIR, index_name: str = FAISS_INDEX_NAME,
                 top_k: int = DEFAULT_TOP_K):
        self.persist_dir = persist_dir
        self.index_name = index_name
        self.top_k = top_k
        self._store = None
        self._retriever = None

    def _ensure_loaded(self):
        if self._store is None:
            self._store = get_or_create_vector_store(
                persist_dir=self.persist_dir, index_name=self.index_name
            )
            self._retriever = build_retriever(self._store, top_k=self.top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> Tuple[str, List[Citation], List[RetrievedChunk]]:
        """
        Returns:
            context_string: formatted text block to insert into the LLM prompt
            citations:       structured Citation objects for storage / API response
            chunks:          the raw RetrievedChunk list (useful for debugging/tests)
        """
        try:
            self._ensure_loaded()
        except VectorStoreNotFoundError:
            logger.error("Vector store not found — run ingestion first.")
            return "", [], []

        chunks = self._retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return "", [], []

        context = self._retriever.build_context_string(chunks)
        citations = generate_citations(chunks)
        return context, citations, chunks

    def refresh(self) -> int:
        """Re-run ingestion and reload the in-memory retriever."""
        count = ingest_knowledge_base(persist_dir=self.persist_dir, index_name=self.index_name)
        self._store = None
        self._retriever = None
        return count
