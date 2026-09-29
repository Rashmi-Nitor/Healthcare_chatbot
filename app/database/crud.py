"""
app/database/crud.py

All read/write operations against the conversations/messages/citations
tables. Routes call into here rather than touching the ORM directly.
"""
from typing import Any, List, Optional, Tuple
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import Conversation, Message, Citation as CitationModel
from app.llm.healthcare_assistant import ChatTurn

# NOTE: deliberately NOT importing app.rag.citation_generator.Citation here.
# That import pulls in the full LangChain/FAISS stack, which the database
# layer has no business depending on just for a type hint. `citations`
# below is duck-typed (reads .source/.page/.chunk_id/.relevance_score/
# .snippet off whatever object is passed) the same way
# guardrails/response_filter.py handles it.


def create_conversation(db: Session) -> Conversation:
    conv = Conversation()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def list_conversations(db: Session) -> List[Tuple[Conversation, int]]:
    message_counts = (
        db.query(
            Message.conversation_id,
            func.count(Message.id).label("message_count"),
        )
        .group_by(Message.conversation_id)
        .subquery()
    )
    return (
        db.query(Conversation, func.coalesce(message_counts.c.message_count, 0))
        .outerjoin(message_counts, Conversation.id == message_counts.c.conversation_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def delete_conversation(db: Session, conversation_id: str) -> bool:
    conv = get_conversation(db, conversation_id)
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True


def add_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    grounded: bool = True,
    blocked_reason: Optional[str] = None,
    citations: Optional[List[Any]] = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        grounded=grounded,
        blocked_reason=blocked_reason,
    )
    db.add(msg)
    db.flush()  # get msg.id before commit

    if citations:
        for c in citations:
            db.add(CitationModel(
                message_id=msg.id,
                source=c.source,
                page=c.page,
                chunk_id=c.chunk_id,
                relevance_score=c.relevance_score,
                snippet=c.snippet,
            ))

    db.commit()
    db.refresh(msg)
    return msg


def get_history_as_chat_turns(conversation: Conversation) -> List[ChatTurn]:
    """Builds the `history` argument HealthcareAssistant.answer() expects."""
    return [ChatTurn(role=m.role, content=m.content) for m in conversation.messages]


def touch_conversation(db: Session, conversation: Conversation) -> None:
    """Bumps updated_at so conversation lists sort by recency."""
    db.add(conversation)
    db.commit()
