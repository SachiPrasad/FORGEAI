"""Pydantic v2 schemas for Feature 5 — Tool Registry & Tool Selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolPriority = Literal["essential", "recommended", "optional"]
ToolCategory = Literal[
    "knowledge_retrieval",
    "computation",
    "structured_data",
    "temporal",
    "external_api",
    "web_search",
    "custom",
]


class ToolRegistryEntry(BaseModel):
    """Metadata describing a tool registered in FORGEAI's central ToolRegistry."""

    tool_id: str = Field(..., min_length=1)
    display_name: str
    category: ToolCategory
    priority: ToolPriority
    purpose: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    requires_knowledge_base: bool = False
    requires_csv_files: bool = False
    requires_network: bool = False
    safe_for_closed_book: bool = True


class ToolDecision(BaseModel):
    """Selection decision and explicit justification for a single registered tool."""

    tool_id: str
    display_name: str
    priority: ToolPriority
    selected: bool
    reason: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def format_line(self) -> str:
        flag = "Yes" if self.selected else "No"
        return f"Tool: {self.display_name} | Selected: {flag} | Reason: {self.reason}"


class LLMToolSelectionBatch(BaseModel):
    """Structured LLM tool selection response."""

    decisions: list[ToolDecision] = Field(default_factory=list)


class ToolSelectionRecord(BaseModel):
    """Audit record of tools dynamically selected (and rejected) for an agent."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    selected_tools: list[str] = Field(default_factory=list)
    rejected_tools: list[str] = Field(default_factory=list)
    decisions: list[ToolDecision] = Field(default_factory=list)
    formatted_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRegistryExport(BaseModel):
    """Serializable snapshot of all tools available in the centralized registry."""

    exported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_tools: int
    tools: list[ToolRegistryEntry] = Field(default_factory=list)

