"""
app/llm/llm_factory.py

Phase 4 (part 1) — LLM integration via OpenRouter.

OpenRouter exposes an OpenAI-compatible /chat/completions API that can serve
Gemini, GPT, Claude, Llama, etc. behind a single key, so we use LangChain's
`ChatOpenAI` client pointed at OpenRouter's base URL instead of a
provider-specific SDK. Swapping the underlying model is then just an env
var change (OPENROUTER_MODEL_NAME) — no code change required.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.constants import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_NAME_DEFAULT,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Raised when OPENROUTER_API_KEY isn't set in the environment."""


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Return a process-wide singleton chat model instance, routed through
    OpenRouter. Reads:
      - OPENROUTER_API_KEY  (required)
      - OPENROUTER_MODEL_NAME (optional, defaults to a Gemini model on OpenRouter)
      - OPENROUTER_SITE_URL / OPENROUTER_SITE_NAME (optional, OpenRouter's
        recommended attribution headers — see https://openrouter.ai/docs)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "Set OPENROUTER_API_KEY in the environment / .env file. "
            "Get a key at https://openrouter.ai/keys"
        )

    model_name = os.getenv("OPENROUTER_MODEL_NAME", OPENROUTER_MODEL_NAME_DEFAULT)

    default_headers = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        default_headers["HTTP-Referer"] = site_url
    if site_name:
        default_headers["X-Title"] = site_name

    logger.info("Initializing OpenRouter model '%s' via %s.", model_name, OPENROUTER_BASE_URL)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        default_headers=default_headers or None,
    )


def reset_llm_cache() -> None:
    """Useful in tests when swapping API keys / models between test cases."""
    get_llm.cache_clear()
