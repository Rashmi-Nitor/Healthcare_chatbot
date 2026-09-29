"""
app/database/schemas.py

Pydantic request/response models for the API layer.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class CitationOut(BaseModel):
    source: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None
    snippet: Optional[str] = None

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None   # None -> a new conversation is created
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    citations: List[CitationOut] = []
    grounded: bool
    blocked_reason: Optional[str] = None
    created_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    grounded: bool
    blocked_reason: Optional[str] = None
    citations: List[CitationOut] = []
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut]
