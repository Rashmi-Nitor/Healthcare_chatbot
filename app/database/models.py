"""
app/database/models.py

SQLAlchemy ORM models.

conversations 1--N messages 1--N citations
(citations belong to a message, not a document — the "document" a
citation points to is just a filename string from the RAG layer,
not a row we manage here, since document ingestion lives in
app/rag/rag_pipeline.py, not this API.)
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "Message", back_populates="conversation",
        cascade="all, delete-orphan", order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)                  # "user" | "assistant"
    content = Column(Text, nullable=False)
    grounded = Column(Boolean, default=True)
    blocked_reason = Column(String, nullable=True)          # "emergency" | "injection" | "medical" | "out_of_scope" | None
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship(
        "Citation", back_populates="message", cascade="all, delete-orphan"
    )


class Citation(Base):
    __tablename__ = "citations"

    id = Column(String, primary_key=True, default=_uuid)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False)
    source = Column(String, nullable=False)                  # filename, e.g. "Common_Disease_FAQ.pdf"
    page = Column(Integer, nullable=True)
    chunk_id = Column(String, nullable=True)
    relevance_score = Column(Float, nullable=True)
    snippet = Column(Text, nullable=True)

    message = relationship("Message", back_populates="citations")
