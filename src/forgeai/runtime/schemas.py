"""Pydantic v2 schemas for Feature 10 — Generated Agent Runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    """Record of a single tool invocation during a runtime turn."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""
    success: bool = True
    error: str | None = None


class ChatTurn(BaseModel):
    """Single message stored in conversation memory."""

    role: Literal["user", "assistant"]
    content: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RuntimeReply(BaseModel):
    """Structured response returned by AgentRuntime.chat()."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    formatted_citations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "unsupported"] = "medium"
    refused_unsupported_topic: bool = False
    mode_used: Literal["live_llm", "deterministic_grounded", "deterministic_fallback", "safety_guard"] = "deterministic_grounded"
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
    error: str | None = None

