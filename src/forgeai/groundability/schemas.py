"""Pydantic v2 schemas for Feature 3 — Groundability Guard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

CoverageStatus = Literal["supported", "partial", "unsupported"]
DocumentSupportLabel = Literal["Available", "Partial", "Missing"]


class TopicAssessment(BaseModel):
    """Groundability assessment for a single requested topic."""

    topic: str
    document_support: DocumentSupportLabel
    status: CoverageStatus
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    supporting_documents: list[str] = Field(default_factory=list)
    evidence_citations: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    reason: str


class LLMTopicVerdict(BaseModel):
    """Structured LLM verification verdict for a single topic."""

    topic: str
    status: CoverageStatus
    reason: str


class LLMGroundabilityBatch(BaseModel):
    """Structured batch of LLM topic groundability verdicts."""

    verdicts: list[LLMTopicVerdict] = Field(default_factory=list)


class GroundabilityReport(BaseModel):
    """Complete groundability evaluation report across all requested topics."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    domain: str
    total_topics_evaluated: int
    groundability_score: float = Field(..., ge=0.0, le=1.0)
    supported_topics: list[str] = Field(default_factory=list)
    partially_supported_topics: list[str] = Field(default_factory=list)
    unsupported_topics: list[str] = Field(default_factory=list)
    topic_assessments: list[TopicAssessment] = Field(default_factory=list)
    generated_restrictions: list[str] = Field(default_factory=list)
    recommendation: str
    available_documents: list[str] = Field(default_factory=list)
    formatted_report: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportedTopicConfig(BaseModel):
    """Safe, grounded topic configuration exported for downstream agent generation."""

    agent_name: str
    domain: str
    target_audience: str
    purpose: str
    tone: str
    output_format: str
    supported_topics: list[str] = Field(default_factory=list)
    partially_supported_topics: list[str] = Field(default_factory=list)
    unsupported_topics: list[str] = Field(default_factory=list)
    allow_partial_topics: bool = True
    partial_topic_caveats: dict[str, str] = Field(default_factory=dict)
    restrictions: list[str] = Field(default_factory=list)
    refusal_policy: str
    allowed_sources: list[str] = Field(default_factory=list)
    groundability_score: float = Field(..., ge=0.0, le=1.0)
    recommendation: str

