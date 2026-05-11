"""Request / response models for the chat API."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

Domain = Literal["canvas", "internal", "general"]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    role: str | None = Field(
        default=None,
        description="Canvas role filter for retrieval (student/instructor/admin/observer)",
    )
    force_domain: Domain | None = Field(
        default=None,
        description="Explicit domain override from UI selection",
    )
    top_k: int = Field(default=15, ge=1, le=30)
    session_id: str | None = Field(
        default=None,
        description="Conversation session ID for multi-turn chat. Omit to start a new session.",
    )


class SourceRef(BaseModel):
    title: str | None
    source_url: str
    score: float
    source_type: Literal["rag", "web"] = "rag"


class ChatResponse(BaseModel):
    answer: str
    domain: Domain
    sources: list[SourceRef]
    matched_keywords: list[str]
    session_id: str | None = None
