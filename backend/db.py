from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from sqlalchemy import (create_engine, Column, Integer, String, DateTime, Text, ForeignKey, JSON)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_URL = os.getenv("SHIKHA_DB_URL", "sqlite:///shikha.db")
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    handle = Column(String(64), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200), default="New Session")
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String(16))  # 'user' | 'assistant'
    text = Column(Text)
    meta = Column(JSON, default={})  # stores retrieval chunks, style, reward, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    reward = Column(Integer)  # +1 / 0 / -1
    note = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

class CorpusDoc(Base):
    __tablename__ = "corpus_docs"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def get_or_create_user(handle: str):
    db = SessionLocal()
    try:
        u = db.query(User).filter_by(handle=handle).first()
        if not u:
            u = User(handle=handle)
            db.add(u)
            db.commit()
            db.refresh(u)
        return u
    finally:
        db.close()