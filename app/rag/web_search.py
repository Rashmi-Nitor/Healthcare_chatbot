"""DuckDuckGo-backed web search used when local retrieval finds no context."""

from __future__ import annotations

import logging
from typing import List

from app.core.constants import MAX_CONTEXT_CHARS, WEB_SEARCH_MAX_RESULTS
from app.rag.citation_generator import Citation

logger = logging.getLogger(__name__)


def search_web(query: str) -> List[Citation]:
    """Return web results as citations; failures leave the caller's fallback intact."""
    if not query or not query.strip():
        return []

    try:
        from ddgs import DDGS

        results = DDGS().text(query.strip(), max_results=WEB_SEARCH_MAX_RESULTS)
        citations = []
        seen_urls = set()
        for result in results:
            url = (result.get("href") or result.get("url") or "").strip()
            title = (result.get("title") or "").strip()
            snippet = (result.get("body") or result.get("snippet") or "").strip()
            if not snippet or (not title and not url):
                continue
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            source = f"{title} ({url})" if title and url else title or url
            citations.append(
                Citation(
                    source=source,
                    page=None,
                    chunk_id=url or None,
                    relevance_score=0.0,
                    snippet=snippet,
                )
            )
        return citations
    except Exception:
        logger.exception("Web search failed; continuing without web results.")
        return []


def build_web_context(citations: List[Citation]) -> str:
    """Format web citations for the LLM within the configured context limit."""
    blocks = [
        f"[Source: {citation.source}]\n{citation.snippet}"
        for citation in citations
        if citation.snippet
    ]
    context = "\n\n---\n\n".join(blocks)
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    return context[:MAX_CONTEXT_CHARS]