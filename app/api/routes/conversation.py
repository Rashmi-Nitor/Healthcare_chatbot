"""
app/api/routes/conversation.py

GET /conversations, GET /conversations/{id}, DELETE /conversations/{id}
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.schemas import ConversationSummary, ConversationDetail, MessageOut, CitationOut
from app.database import crud

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(db: Session = Depends(get_db)):
    convs = crud.list_conversations(db)
    return [
        ConversationSummary(
            id=c.id, title=c.title, created_at=c.created_at,
            updated_at=c.updated_at, message_count=message_count,
        ) for c, message_count in convs
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation_detail(conversation_id: str, db: Session = Depends(get_db)):
    conv = crud.get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(
        id=conv.id, title=conv.title, created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageOut(
                id=m.id, role=m.role, content=m.content,
                grounded=m.grounded, blocked_reason=m.blocked_reason,
                citations=[
                    CitationOut(
                        source=c.source, page=c.page,
                        relevance_score=c.relevance_score, snippet=c.snippet,
                    ) for c in m.citations
                ],
                created_at=m.created_at,
            ) for m in conv.messages
        ],
    )


@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, db: Session = Depends(get_db)):
    ok = crud.delete_conversation(db, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}
