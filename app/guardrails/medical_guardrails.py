"""
app/guardrails/medical_guardrails.py

Blocks requests that ask the assistant to diagnose a condition, interpret
symptoms as a diagnosis, prescribe medication, or recommend dosages /
treatment plans. This runs on the INCOMING message (pre-LLM) so we never
even attempt to answer these — regardless of what the knowledge base
contains.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from app.core.constants import DIAGNOSIS_PATTERNS, DIAGNOSIS_REFUSAL_MSG

logger = logging.getLogger(__name__)

_COMPILED = [re.compile(p, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS]


@dataclass
class MedicalGuardrailResult:
    is_blocked: bool
    matched_patterns: List[str]
    message: str | None = None


def check_medical_guardrails(text: str) -> MedicalGuardrailResult:
    """
    Return a MedicalGuardrailResult. If `is_blocked` is True, respond with
    `message` instead of running retrieval/LLM generation.
    """
    if not text:
        return MedicalGuardrailResult(is_blocked=False, matched_patterns=[])

    matched = [p for p, compiled in zip(DIAGNOSIS_PATTERNS, _COMPILED) if compiled.search(text)]

    if matched:
        logger.warning("Medical guardrail triggered. Patterns: %s", matched)
        return MedicalGuardrailResult(is_blocked=True, matched_patterns=matched, message=DIAGNOSIS_REFUSAL_MSG)

    return MedicalGuardrailResult(is_blocked=False, matched_patterns=[])
