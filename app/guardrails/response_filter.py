"""
app/guardrails/response_filter.py

Post-generation guardrail: runs AFTER the LLM produces an answer, as a
second line of defense. Even with a strict system prompt, models can slip
and use diagnostic/prescriptive language — this catches and neutralizes
that, and makes sure a "Sources:" line is present whenever citations exist.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from app.core.constants import DIAGNOSIS_REFUSAL_MSG
from app.rag.citation_generator import Citation, format_sources_line

logger = logging.getLogger(__name__)

# Phrases that would indicate the model slipped into diagnostic/prescriptive
# territory despite instructions.
_UNSAFE_PATTERNS = [
    r"\byou (have|likely have|probably have)\b",
    r"\byou (are|might be) suffering from\b",
    r"\bi diagnose\b",
    r"\btake \d+\s?(mg|milligrams|pills|tablets)\b",
    r"\byou should take\b.*\b(mg|milligrams|dose|dosage)\b",
    r"\bi recommend (taking|you take)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]


@dataclass
class FilterResult:
    text: str
    was_modified: bool
    flagged_patterns: List[str]


def filter_response(raw_answer: str) -> FilterResult:
    """
    Scan the model's raw answer for diagnostic/prescriptive language. If
    found, replace the entire answer with the standard refusal rather than
    trying to surgically edit it (safer than partial redaction).
    """
    if not raw_answer:
        return FilterResult(text=raw_answer, was_modified=False, flagged_patterns=[])

    flagged = [p for p, compiled in zip(_UNSAFE_PATTERNS, _COMPILED) if compiled.search(raw_answer)]

    if flagged:
        logger.warning("Post-generation filter caught unsafe language: %s", flagged)
        return FilterResult(text=DIAGNOSIS_REFUSAL_MSG, was_modified=True, flagged_patterns=flagged)

    return FilterResult(text=raw_answer, was_modified=False, flagged_patterns=[])


def ensure_sources_appended(answer: str, citations: List[Citation]) -> str:
    """
    Guarantee a 'Sources:' line is present whenever we actually have
    citations, even if the LLM forgot to add one itself.
    """
    if not citations:
        return answer
    if "sources:" in answer.lower():
        return answer
    return f"{answer.rstrip()}\n\n{format_sources_line(citations)}"
