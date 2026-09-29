"""
app/api/routes/chat.py

POST /chat — the main endpoint. Persists the user message, runs the
LangGraph-based HealthcareAssistant (guardrails + retrieval +
generation all happen inside .answer()), persists the assistant
message + citations, and returns both to the caller.
"""
from functools import lru_cache

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.schemas import ChatRequest, ChatResponse, CitationOut
from app.database import crud
from app.llm.healthcare_assistant import HealthcareAssistant

router = APIRouter(tags=["chat"])


@lru_cache(maxsize=1)
def get_assistant() -> HealthcareAssistant:
    """
    Process-wide singleton — building the LangGraph app and loading the
    embedding model on every request would be far too slow. Cached the
    same way llm_factory.get_llm() caches the chat model.
    """
    return HealthcareAssistant()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    conversation = (
        crud.get_conversation(db, request.conversation_id)
        if request.conversation_id else None
    )
    if conversation is None:
        conversation = crud.create_conversation(db)

    history = crud.get_history_as_chat_turns(conversation)

    crud.add_message(db, conversation.id, role="user", content=request.message)

    assistant = get_assistant()
    result = assistant.answer(request.message, history=history)

    assistant_msg = crud.add_message(
        db, conversation.id, role="assistant", content=result.answer,
        grounded=result.grounded, blocked_reason=result.blocked_reason,
        citations=result.citations,
    )
    crud.touch_conversation(db, conversation)

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=assistant_msg.id,
        answer=assistant_msg.content,
        citations=[
            CitationOut(
                source=c.source, page=c.page,
                relevance_score=c.relevance_score, snippet=c.snippet,
            ) for c in result.citations
        ],
        grounded=assistant_msg.grounded,
        blocked_reason=assistant_msg.blocked_reason,
        created_at=assistant_msg.created_at,
    )
