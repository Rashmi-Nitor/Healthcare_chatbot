"""
app/llm/healthcare_assistant.py

The conversational orchestrator — now implemented as a LangGraph
`StateGraph` instead of a linear Python function chain.

The graph runs retrieval, generation, and response finalization. Generation
errors are routed to the end without running later stages.

Graph topology:

    START
      │
      ▼
  retrieve -> generate -> finalize -> END
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from app.core.constants import NO_ANSWER_FOUND_MSG
from app.core.prompts import SYSTEM_PROMPT, RAG_ANSWER_TEMPLATE
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk
from app.rag.citation_generator import Citation
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

    # populated by `retrieve`
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
# Node functions
# ---------------------------------------------------------------------------
def _make_retrieve_node(rag: RAGPipeline):
    def node_retrieve(state: AssistantState) -> AssistantState:
        context, citations, chunks = rag.retrieve(state["question"])
        return {"context": context, "citations": citations, "chunks": chunks}
    return node_retrieve


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
    graph.add_node("generate", node_generate)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
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