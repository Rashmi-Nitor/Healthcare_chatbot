"""
app/rag/text_splitter.py

Phase 2 (part 1) — Text chunking.

Wraps LangChain's RecursiveCharacterTextSplitter with our project defaults
and makes sure chunk-level metadata (chunk_id, source, page) is preserved
and extended so citations can later point back to an exact chunk.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.constants import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _make_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Deterministic short hash so re-ingesting the same content is idempotent."""
    digest = hashlib.sha1(f"{source}:{page}:{chunk_index}:{text[:50]}".encode()).hexdigest()
    return digest[:12]


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks suitable for
    embedding. Each resulting chunk keeps the parent's metadata plus a
    unique `chunk_id` and `chunk_index`.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )

    chunks: List[Document] = []
    for doc in documents:
        pieces = splitter.split_text(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)

        for idx, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            meta["chunk_id"] = _make_chunk_id(source, page, idx, piece)
            chunks.append(Document(page_content=piece, metadata=meta))

    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks
