from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    dominant_emotion: str
    emotion_scores: dict
    situation: str
    strategy: str
    flat_affect_flag: bool
    trained_model_used: bool
    retrieved_memories: list[str]
    memory_mode: str
    tool_calls: int
    critique_issues: list[str]
    regenerated: bool
    reflection: dict
    user_message_id: str
    assistant_message_id: str


class ProfileImportRequest(BaseModel):
    user_id: str
    text: str


class FeedbackRequest(BaseModel):
    user_id: str
    message_id: str
    rating: str  # "up" | "down"
    tags: list[str] = []


class CreateUserResponse(BaseModel):
    user_id: str
