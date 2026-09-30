"""
app/guardrails/emergency_detector.py

Scans incoming user messages for emergency-adjacent language (chest pain,
stroke symptoms, suicidal ideation, etc.) and decides between two very
different cases that both mention the same keywords:

    1. "I'm having chest pain right now, help me"   -> REAL emergency.
       Bypass RAG/LLM entirely and return a safety message immediately.

    2. "What are the symptoms of a heart attack?"   -> informational
       question. Should be answered normally from the knowledge base — we
       just attach a light safety reminder to the final answer.

Getting this distinction wrong in the "block everything" direction makes
the assistant useless for the exact patient-education use case it exists
for; getting it wrong in the "allow everything" direction risks failing to
flag a real emergency. So the detector is layered, with safety-first
tie-breaking on genuinely ambiguous input:

    keyword hit?
        no  -> not an emergency
        yes -> distress signal present?      ("I'm having...", "help me",
                                               "call 911", first-person +
                                               present tense)
                yes -> EMERGENCY (regardless of anything else)
                no  -> informational framing present?  ("what is...",
                                                         "symptoms of...",
                                                         "tell me about...")
                        yes -> NOT an emergency (informational)
                        no  -> genuinely ambiguous -> ask the LLM to
                               classify EXPERIENCING vs ASKING; on any
                               failure, fail SAFE (treat as emergency).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.constants import (
    EMERGENCY_KEYWORDS,
    EMERGENCY_DISTRESS_PATTERNS,
    EMERGENCY_INFORMATIONAL_PATTERNS,
    EMERGENCY_MSG,
    EMERGENCY_INFO_SAFETY_NOTE,
)

logger = logging.getLogger(__name__)

_KEYWORD_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in EMERGENCY_KEYWORDS]
_DISTRESS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_DISTRESS_PATTERNS]
_INFO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_INFORMATIONAL_PATTERNS]


@dataclass
class EmergencyCheckResult:
    is_emergency: bool
    matched_terms: List[str]
    message: Optional[str] = None
    # True when the message mentions an emergency-adjacent topic but reads
    # as an educational question rather than a live emergency.
    is_informational: bool = False
    # Light reminder to append to the normal RAG answer when is_informational.
    safety_note: Optional[str] = None


def _has_distress_signal(text: str) -> bool:
    return any(p.search(text) for p in _DISTRESS_PATTERNS)


def _has_informational_signal(text: str) -> bool:
    return any(p.search(text) for p in _INFO_PATTERNS)


def check_emergency(text: str, use_llm_fallback: bool = True) -> EmergencyCheckResult:
    """
    Return an EmergencyCheckResult.
      - If `is_emergency` is True: callers MUST short-circuit and return
        `message` directly, without invoking retrieval or the LLM.
      - If `is_informational` is True: callers should proceed with the
        normal RAG flow, and append `safety_note` to the final answer.
    """
    if not text:
        return EmergencyCheckResult(is_emergency=False, matched_terms=[])

    matched_keywords = [
        kw for kw, pattern in zip(EMERGENCY_KEYWORDS, _KEYWORD_PATTERNS) if pattern.search(text)
    ]
    if not matched_keywords:
        return EmergencyCheckResult(is_emergency=False, matched_terms=[])

    # 1. Distress language wins outright — safety first, no matter what else
    #    is in the message.
    if _has_distress_signal(text):
        logger.warning("Emergency (distress signal) detected: %s", matched_keywords)
        return EmergencyCheckResult(is_emergency=True, matched_terms=matched_keywords, message=EMERGENCY_MSG)

    # 2. Clear educational framing and no distress signal -> informational.
    if _has_informational_signal(text):
        logger.info("Emergency-adjacent topic asked informationally: %s", matched_keywords)
        return EmergencyCheckResult(
            is_emergency=False,
            matched_terms=matched_keywords,
            is_informational=True,
            safety_note=EMERGENCY_INFO_SAFETY_NOTE,
        )

    # 3. Genuinely ambiguous (e.g. just "heart attack" with no other
    #    context) -> let the LLM classify; fail SAFE on any error.
    if not use_llm_fallback:
        logger.warning("Ambiguous emergency phrasing, no LLM fallback — failing safe: %s", matched_keywords)
        return EmergencyCheckResult(is_emergency=True, matched_terms=matched_keywords, message=EMERGENCY_MSG)

    return _llm_fallback_check(text, matched_keywords)


def _llm_fallback_check(text: str, matched_keywords: List[str]) -> EmergencyCheckResult:
    """Single strict classification call for genuinely ambiguous messages."""
    try:
        from app.llm.llm_factory import get_llm  # local import avoids a hard dependency at module load

        llm = get_llm()
        prompt = (
            "Answer with exactly one word: EXPERIENCING or ASKING.\n"
            "EXPERIENCING = the user is describing a medical emergency they "
            "(or someone with them) are going through right now and need "
            "urgent help.\n"
            "ASKING = the user is asking an informational or educational "
            "question about a medical topic, not reporting a current "
            "emergency.\n\n"
            f"Message: {text}\n\nAnswer:"
        )
        response = llm.invoke(prompt)
        answer = (response.content or "").strip().upper()
        logger.info("LLM emergency classification for %r -> %s", text, answer)

        if answer.startswith("EXPERIENCING"):
            return EmergencyCheckResult(is_emergency=True, matched_terms=matched_keywords, message=EMERGENCY_MSG)

        return EmergencyCheckResult(
            is_emergency=False,
            matched_terms=matched_keywords,
            is_informational=True,
            safety_note=EMERGENCY_INFO_SAFETY_NOTE,
        )
    except Exception:
        logger.exception("LLM fallback emergency check failed; failing SAFE (treating as emergency).")
        # Unlike topic_validator (which fails OPEN toward retrieval), this
        # guardrail fails CLOSED toward the emergency message — the cost of
        # a false positive here is much lower than a missed real emergency.
        return EmergencyCheckResult(is_emergency=True, matched_terms=matched_keywords, message=EMERGENCY_MSG)