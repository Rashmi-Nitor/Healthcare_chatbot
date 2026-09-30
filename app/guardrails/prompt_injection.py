"""
app/guardrails/prompt_injection.py

Detects attempts to override the assistant's configured role/system prompt
via the user's message — e.g. "ignore previous instructions", "reveal your
system prompt", "act as my doctor", "forget the knowledge base". Runs before
the message is ever included in an LLM prompt.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from app.core.constants import PROMPT_INJECTION_PATTERNS, PROMPT_INJECTION_MSG

logger = logging.getLogger(__name__)

_COMPILED = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]


@dataclass
class InjectionCheckResult:
    is_injection: bool
    matched_patterns: List[str]
    message: str | None = None


def check_prompt_injection(text: str) -> InjectionCheckResult:
    """
    Return an InjectionCheckResult. If `is_injection` is True, the caller
    should refuse the request with `message` instead of forwarding it to the
    RAG pipeline / LLM.
    """
    if not text:
        return InjectionCheckResult(is_injection=False, matched_patterns=[])

    matched = [p for p, compiled in zip(PROMPT_INJECTION_PATTERNS, _COMPILED) if compiled.search(text)]

    if matched:
        logger.warning("Prompt injection attempt detected. Patterns: %s", matched)
        return InjectionCheckResult(is_injection=True, matched_patterns=matched, message=PROMPT_INJECTION_MSG)

    return InjectionCheckResult(is_injection=False, matched_patterns=[])
