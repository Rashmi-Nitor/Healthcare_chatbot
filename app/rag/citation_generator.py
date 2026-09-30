"""
app/rag/citation_generator.py

Phase 5 — Source citation generation.

Turns the list of RetrievedChunk objects that were actually fed to the LLM
into structured citation records (for storage / the API response), and
provides a helper to format a human-readable "Sources:" line, e.g.:

    According to Patient_Care_Guidelines.pdf...
    Sources: Patient_Care_Guidelines.pdf (p. 3), Common_Disease_FAQ.pdf (p. 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List

from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    source: str
    page: int | None
    chunk_id: str | None
    relevance_score: float
    snippet: str  # short preview of the chunk text, for UI display

    def to_dict(self) -> dict:
        return asdict(self)


def generate_citations(chunks: List[RetrievedChunk], snippet_len: int = 160) -> List[Citation]:
    """
    Build one Citation per retrieved chunk (deduplicated by source+page),
    preserving the highest-scoring occurrence.
    """
    seen: dict[tuple, Citation] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        text = chunk.document.page_content.strip().replace("\n", " ")
        snippet = (text[:snippet_len] + "...") if len(text) > snippet_len else text

        citation = Citation(
            source=chunk.source,
            page=chunk.page,
            chunk_id=chunk.document.metadata.get("chunk_id"),
            relevance_score=round(chunk.score, 4),
            snippet=snippet,
        )
        # Keep whichever occurrence has the higher relevance score.
        if key not in seen or citation.relevance_score > seen[key].relevance_score:
            seen[key] = citation

    citations = sorted(seen.values(), key=lambda c: c.relevance_score, reverse=True)
    logger.info("Generated %d citation(s) from %d retrieved chunk(s).",
                len(citations), len(chunks))
    return citations


def format_sources_line(citations: List[Citation]) -> str:
    """Human-readable 'Sources: ...' line for appending to an answer."""
    if not citations:
        return ""
    parts = []
    for c in citations:
        page_part = f" (p. {c.page})" if c.page is not None else ""
        parts.append(f"{c.source}{page_part}")
    return "Sources: " + ", ".join(parts)


def unique_source_filenames(citations: List[Citation]) -> List[str]:
    """Distinct list of source document filenames, in relevance order."""
    seen = set()
    ordered = []
    for c in citations:
        if c.source not in seen:
            seen.add(c.source)
            ordered.append(c.source)
    return ordered
