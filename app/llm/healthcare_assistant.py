"""
app/llm/healthcare_assistant.py

The conversational orchestrator — now implemented as a LangGraph
`StateGraph` instead of a linear Python function chain.

The graph runs retrieval, an optional web-search fallback (when the local
knowledge base has no relevant context), generation, and response
finalization. Generation errors are routed to the end without running
later stages.

Graph topology:

    START
      │
      ▼
   retrieve ──(no local context AND looks like a real health question)──▶ web_search_fallback ──┐
      │                                                                                             │
      │ (has local context, OR greeting/off-topic/too-short)                                       ▼
      └──────────────────────────────────────────────────────────────────────────────────▶ generate ──(failed)──▶ END
                                                                                                    │
                                                                                                    ▼ (continue)
                                                                                                finalize ──▶ END
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from app.core.constants import NO_ANSWER_FOUND_MSG
from app.core.prompts import SYSTEM_PROMPT, RAG_ANSWER_TEMPLATE
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk
from app.rag.citation_generator import Citation
from app.rag.web_search import search_web, build_web_context
from app.llm.llm_factory import get_llm
from app.guardrails.response_filter import ensure_sources_appended

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class AssistantResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    blocked_reason: Optional[str] = None  # "emergency" | "medical" | "injection" | "out_of_scope" | None
    grounded: bool = True  # False when we fell back / were blocked / were filtered


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AssistantState(TypedDict, total=False):
    """
    The shared state object every node reads from and writes into.
    `total=False` because most keys are only populated once their node runs.
    """
    question: str
    history: List[ChatTurn]

    # populated when generation fails
    blocked_reason: Optional[str]
    blocked_message: Optional[str]

    # populated by `retrieve`, and possibly overwritten by `web_search_fallback`
    context: str
    citations: List[Citation]
    chunks: List[RetrievedChunk]

    # populated by `generate` / `post_filter`
    raw_answer: str
    final_answer: str
    grounded: bool


def _format_history(history: List[ChatTurn], max_turns: int = 6) -> str:
    if not history:
        return "(no previous conversation)"
    recent = history[-max_turns:]
    return "\n".join(f"{turn.role.capitalize()}: {turn.content}" for turn in recent)


# ---------------------------------------------------------------------------
# Web-search gate
# ---------------------------------------------------------------------------
# Keeps DuckDuckGo (and therefore citations) out of greetings, off-topic
# requests, and other very-short replies — i.e. the cases prompts.py calls
# Situation C/D — which have empty local `context` for the same reason a
# genuine Situation B health question does (the KB is medical PDFs, so
# neither matches), but should never end up with a "Sources:" line attached
# to a canned, uncited reply.
_HEALTH_HINT_PATTERN = re.compile(
    r"\b(health|medical|medicine|disease|diagnos|symptom|treatment|doctor|"
    r"physician|patient|hospital|clinic|condition|disorder|syndrome|therapy|"
    r"medication|drug|insurance|cardio|heart|blood|pressure|cholesterol|"
    r"diabetes|mental|anxiety|depression|infection|virus|bacteria|immune|"
    r"vaccine|cancer|chronic|acute|prescription|surgery|nurse|wellness|"
    r"nutrition|diet|sleep|exercise|pain|fever|allergy|pregnan|vitamin|"
    r"bmi|obesity|stroke|asthma|arthritis)\b",
    re.IGNORECASE,
)

_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|good (morning|afternoon|evening|night)|"
    r"thanks|thank you|thx|ok|okay|bye|goodbye|cool|great|nice)[\s!.,]*$",
    re.IGNORECASE,
)

_OFF_TOPIC_HINT_PATTERN = re.compile(
    r"\b(joke|write (a |some )?(python|java|javascript|code)|who won|ipl|"
    r"weather|stock price|poem|movie|song|cricket score|football score)\b",
    re.IGNORECASE,
)


def _should_try_web_search(question: str) -> bool:
    """
    True only for messages that look like a genuine, on-topic health
    question that simply isn't covered by the local KB. False for
    greetings/small talk, clearly off-topic requests, and very short
    non-health replies — so those fall straight through to `generate` with
    no citations to (wrongly) attach.
    """
    text = question.strip()
    if not text:
        return False
    if _GREETING_PATTERN.match(text):
        return False
    if _OFF_TOPIC_HINT_PATTERN.search(text) and not _HEALTH_HINT_PATTERN.search(text):
        return False
    if len(text.split()) <= 3 and not _HEALTH_HINT_PATTERN.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def _make_retrieve_node(rag: RAGPipeline):
    def node_retrieve(state: AssistantState) -> AssistantState:
        context, citations, chunks = rag.retrieve(state["question"])
        return {"context": context, "citations": citations, "chunks": chunks}
    return node_retrieve


def node_web_search_fallback(state: AssistantState) -> AssistantState:
    """
    Runs only when `retrieve` found no relevant local context (Situation B
    in prompts.py). Tries DuckDuckGo before falling back to the LLM's own
    parametric knowledge, so Situation B answers can carry real citations
    when the web has something relevant.

    Never blocks the flow: if DuckDuckGo also finds nothing (or errors),
    this returns {} and `generate` proceeds with empty context exactly as
    it did before this node existed — the system prompt's Situation B rule
    (answer from general knowledge, with the "not present in the data"
    disclaimer) still applies.
    """
    citations = search_web(state["question"])
    if not citations:
        return {}
    return {
        "context": build_web_context(citations),
        "citations": citations,
    }


def node_generate(state: AssistantState) -> AssistantState:
    prompt = RAG_ANSWER_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        chat_history=_format_history(state.get("history", [])),
        context=state["context"],
        question=state["question"],
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_answer = response.content or ""
    except Exception:
        logger.exception("[graph] generate node: LLM call failed.")
        return {
            "raw_answer": "",
            "blocked_reason": "llm_error",
            "blocked_message": "Something went wrong while generating a response. Please try again shortly.",
        }
    return {"raw_answer": raw_answer}


def node_finalize(state: AssistantState) -> AssistantState:
    final_answer = ensure_sources_appended(state["raw_answer"], state.get("citations", []))
    return {
        "final_answer": final_answer,
        "grounded": bool(state.get("context")),
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def _route_after_retrieve(state: AssistantState) -> Literal["needs_web", "skip_web"]:
    if state.get("context"):
        return "skip_web"  # local KB already answered it — no web search needed
    if _should_try_web_search(state["question"]):
        return "needs_web"
    return "skip_web"  # greeting / off-topic / too-short — skip DuckDuckGo entirely


def _route_after_generate(state: AssistantState) -> Literal["failed", "continue"]:
    return "failed" if state.get("blocked_reason") else "continue"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(rag_pipeline: Optional[RAGPipeline] = None):
    """Construct and compile the LangGraph StateGraph for the assistant."""
    rag = rag_pipeline or RAGPipeline()
    graph = StateGraph(AssistantState)

    graph.add_node("retrieve", _make_retrieve_node(rag))
    graph.add_node("web_search_fallback", node_web_search_fallback)
    graph.add_node("generate", node_generate)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", _route_after_retrieve,
        {"needs_web": "web_search_fallback", "skip_web": "generate"},
    )
    graph.add_edge("web_search_fallback", "generate")
    graph.add_conditional_edges(
        "generate", _route_after_generate, {"failed": END, "continue": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public wrapper — same call signature as the pre-LangGraph version, so
# Person 2's FastAPI route doesn't need to change at all.
# ---------------------------------------------------------------------------
class HealthcareAssistant:
    """
    Thin, stateless wrapper around the compiled LangGraph app. Conversation
    history is still passed in by the caller (persisted in SQLite by Person
    2's service layer) rather than held here.
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag = rag_pipeline or RAGPipeline()
        self._app = build_graph(self.rag)

    def answer(self, question: str, history: Optional[List[ChatTurn]] = None) -> AssistantResponse:
        initial_state: AssistantState = {
            "question": question,
            "history": history or [],
            "blocked_reason": None,
        }

        final_state: AssistantState = self._app.invoke(initial_state)

        if final_state.get("blocked_reason"):
            reason = final_state["blocked_reason"]
            return AssistantResponse(
                answer=final_state.get("blocked_message", NO_ANSWER_FOUND_MSG),
                citations=final_state.get("citations", []) or [],
                blocked_reason=None,
                grounded=False,
            )

        return AssistantResponse(
            answer=final_state.get("final_answer", NO_ANSWER_FOUND_MSG),
            citations=final_state.get("citations", []) or [],
            blocked_reason=None,
            grounded=final_state.get("grounded", True),
        )