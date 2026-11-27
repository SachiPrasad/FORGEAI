"""Pydantic v2 schemas for Feature 4 — Agent Generator (agent.json)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from forgeai.tools.schemas import ToolSelectionRecord


class AgentModelConfig(BaseModel):
    """LLM configuration and runtime parameters for the generated agent."""

    model: str = Field(..., min_length=1, description="Model identifier (e.g., 'gpt-4o-mini', 'openai/o3').")
    provider: str = Field(default="litellm", description="LLM provider abstraction.")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=128, le=32768)
    max_turns: int = Field(default=10, ge=1, le=50)
    tool_choice: Literal["auto", "required", "none"] = "auto"


class RetrievalConfig(BaseModel):
    """Knowledge retrieval configuration controlling chunking & top-k search."""

    chunk_size: int = Field(default=700, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    top_k: int = Field(default=4, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    hybrid_keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    strategy: str = "recursive"
    knowledge_manifest_path: str | None = None
    vector_store_path: str | None = None

    @model_validator(mode="after")
    def _validate_overlap(self) -> RetrievalConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self


class ToolSpec(BaseModel):
    """Specification of a selected tool made available to the generated agent."""

    name: str
    type: Literal[
        "knowledge_retrieval",
        "computation",
        "structured_data",
        "temporal",
        "external_api",
        "web_search",
        "guardrail",
        "nlp_utility",
        "web_lookup",
        "custom",
    ]
    description: str
    enabled: bool = True
    selection_reason: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentValidationIssue(BaseModel):
    """Single validation finding for a generated AgentDefinition."""

    severity: Literal["error", "warning"]
    component: str
    message: str


class AgentValidationReport(BaseModel):
    """Validation result returned by AgentDefinitionValidator."""

    is_valid: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    checks_passed: list[str] = Field(default_factory=list)
    issues: list[AgentValidationIssue] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    """Complete agent blueprint serialized to agent.json."""

    agent_name: str
    domain: str
    target_audience: str
    purpose: str
    tone: str
    output_format: str
    model: str
    model_config_details: AgentModelConfig = Field(..., alias="model_config")
    system_prompt: str
    retrieval_config: RetrievalConfig
    tools: list[ToolSpec] = Field(default_factory=list)
    rejected_tools: list[dict[str, str]] = Field(default_factory=list)
    tool_selection_record: ToolSelectionRecord | None = None
    supported_topics: list[str] = Field(default_factory=list)
    partially_supported_topics: list[str] = Field(default_factory=list)
    unsupported_topics: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    allowed_sources: list[str] = Field(default_factory=list)
    output_contract: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)
    structured_output_code: str = ""
    validation: AgentValidationReport = Field(
        default_factory=lambda: AgentValidationReport(is_valid=True, completeness_score=1.0)
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {"generated_at": datetime.now(timezone.utc).isoformat()}
    )

    model_config = {"populate_by_name": True}

