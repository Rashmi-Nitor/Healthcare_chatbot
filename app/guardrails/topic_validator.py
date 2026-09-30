"""
app/guardrails/topic_validator.py

Out-of-scope detection: rejects questions unrelated to healthcare
(e.g. "tell me a joke", "write Python code", "who won IPL?").

Strategy (cheap-first, to avoid burning an LLM call on every message):
  1. Fast keyword/heuristic pass using a healthcare vocabulary list.
  2. If the heuristic is inconclusive, fall back to a single lightweight
     LLM classification call ("healthcare" vs "other").

The heuristic pass alone catches the overwhelming majority of both clearly
in-scope and clearly out-of-scope messages, so step 2 is rare in practice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.core.constants import GREETING_MSG, OUT_OF_SCOPE_MSG

logger = logging.getLogger(__name__)

# Broad healthcare vocabulary — deliberately generous so we don't block
# legitimate questions just for using everyday phrasing.
_HEALTHCARE_TERMS = [
    "health", "medical", "medicine", "disease", "diagnos", "symptom",
    "treatment", "doctor", "physician", "patient", "hospital", "clinic",
    "condition", "disorder", "syndrome", "therapy", "medication", "drug",
    "insurance", "cardio", "heart", "blood", "pressure", "cholesterol",
    "diabetes", "mental", "anxiety", "depression", "infection", "virus",
    "bacteria", "immune", "vaccine", "cancer", "chronic", "acute",
    "prescription", "surgery", "nurse", "wellness", "nutrition", "glossary",
    "terminology", "care guideline", "faq", "stroke", "obesity", "metabolic",
]

# Clearly non-healthcare requests we want to bounce quickly.
_OFF_TOPIC_HINTS = [
    r"\bjoke\b", r"\bwrite (a |some )?(python|java|javascript|code)\b",
    r"\bwho won\b", r"\bipl\b", r"\bweather\b", r"\bstock price\b",
    r"\bpoem\b", r"\brecipe for\b(?!.*(health|diet))", r"\bsong\b",
    r"\bmovie\b", r"\bfootball\b", r"\bcricket score\b",
]

_HEALTH_PATTERN = re.compile("|".join(_HEALTHCARE_TERMS), re.IGNORECASE)
_OFF_TOPIC_PATTERN = re.compile("|".join(_OFF_TOPIC_HINTS), re.IGNORECASE)
_GREETING_PATTERN = re.compile(
    r"^(?:hello|hi|hey|greetings|howdy|hiya|good\s+(?:morning|afternoon|evening|night))"
    r"(?:[\s,!?]+(?:there|everyone|chatbot|assistant))?[\s!,.?]*$",
    re.IGNORECASE,
)


@dataclass
class TopicCheckResult:
    in_scope: bool
    confidence: str  # "heuristic" or "llm"
    message: str | None = None


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_PATTERN.fullmatch(text.strip()))


def _heuristic_check(text: str) -> bool | None:
    """Return True (in scope), False (out of scope), or None (inconclusive)."""
    has_health_term = bool(_HEALTH_PATTERN.search(text))
    has_off_topic_hint = bool(_OFF_TOPIC_PATTERN.search(text))

    if has_off_topic_hint and not has_health_term:
        return False
    if has_health_term:
        return True
    return None


def check_topic(text: str, use_llm_fallback: bool = True) -> TopicCheckResult:
    """
    Determine whether `text` is a healthcare-related query.
    If the heuristic is inconclusive and `use_llm_fallback` is True, ask the
    LLM a single strict classification question.
    """
    if not text or not text.strip():
        return TopicCheckResult(in_scope=False, confidence="heuristic", message=OUT_OF_SCOPE_MSG)

    if _is_greeting(text):
        return TopicCheckResult(in_scope=False, confidence="heuristic", message=GREETING_MSG)

    heuristic_result = _heuristic_check(text)
    if heuristic_result is True:
        return TopicCheckResult(in_scope=True, confidence="heuristic")
    if heuristic_result is False:
        logger.info("Out-of-scope message rejected by heuristic: %r", text)
        return TopicCheckResult(in_scope=False, confidence="heuristic", message=OUT_OF_SCOPE_MSG)

    if not use_llm_fallback:
        # Default to allowing it through to retrieval; if nothing relevant is
        # found there, the pipeline's "no answer found" guardrail handles it.
        return TopicCheckResult(in_scope=True, confidence="heuristic")

    return _llm_fallback_check(text)


def _llm_fallback_check(text: str) -> TopicCheckResult:
    """Single strict yes/no LLM call for ambiguous messages."""
    try:
        from app.llm.llm_factory import get_llm  # local import avoids a hard dependency at module load

        llm = get_llm()
        prompt = (
            "Answer with exactly one word, 'YES' or 'NO'. "
            "Is the following user message a healthcare, medical, or "
            "medical-terminology related question?\n\n"
            f"Message: {text}\n\nAnswer:"
        )
        response = llm.invoke(prompt)
        answer = (response.content or "").strip().upper()
        in_scope = answer.startswith("Y")
        logger.info("LLM topic classification for %r -> %s", text, answer)
        if in_scope:
            return TopicCheckResult(in_scope=True, confidence="llm")
        return TopicCheckResult(in_scope=False, confidence="llm", message=OUT_OF_SCOPE_MSG)
    except Exception:
        logger.exception("LLM fallback topic check failed; defaulting to in-scope.")
        # Fail open toward retrieval rather than blocking a possibly-valid question.
        return TopicCheckResult(in_scope=True, confidence="heuristic")
