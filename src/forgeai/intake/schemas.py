"""Pydantic v2 schemas for Feature 1 — Intelligent Intake Interviewer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_CLARIFICATION_QUESTIONS = 5


class UserConfig(BaseModel):
    """Optional user configuration controlling extraction, overrides, and runtime mode."""

    model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM model identifier (e.g., 'gpt-4o-mini', 'openai/o3', 'gemini/gemini-2.5-flash').",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    use_llm: bool | None = Field(
        default=None,
        description="None = auto-detect API keys; True = force LLM; False = deterministic offline extraction.",
    )
    max_clarification_questions: int = Field(
        default=MAX_CLARIFICATION_QUESTIONS,
        ge=1,
        le=MAX_CLARIFICATION_QUESTIONS,
        description="Maximum clarification questions to generate (hard cap: 5).",
    )
    agent_name_override: str | None = None
    domain_override: str | None = None
    target_audience_override: str | None = None
    tone_override: str | None = None
    output_format_override: str | None = None
    extra_restrictions: list[str] = Field(default_factory=list)
    extra_unsupported_topics: list[str] = Field(default_factory=list)
    include_subdirectories: bool = True

    @field_validator("max_clarification_questions", mode="before")
    @classmethod
    def _cap_questions(cls, value: int) -> int:
        if isinstance(value, int) and value > MAX_CLARIFICATION_QUESTIONS:
            return MAX_CLARIFICATION_QUESTIONS
        return value


class ParsedBrief(BaseModel):
    """Structured analysis of the raw business brief."""

    raw_text: str = Field(..., min_length=1)
    normalized_text: str
    word_count: int
    domain_hints: list[str] = Field(default_factory=list)
    audience_hints: list[str] = Field(default_factory=list)
    action_intents: list[str] = Field(default_factory=list)
    candidate_topics: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    tone_hints: list[str] = Field(default_factory=list)
    output_format_hints: list[str] = Field(default_factory=list)
    constraint_hints: list[str] = Field(default_factory=list)
    ambiguity_signals: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    """Extracted metadata and content summary for a single document."""

    filename: str
    relative_path: str
    doc_type: Literal["pdf", "docx", "txt", "csv"]
    source_label: str = Field(..., description="Human-readable label such as 'HR handbook' or 'leave policy'.")
    char_count: int = 0
    word_count: int = 0
    page_count: int | None = None
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    detected_headings: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    excerpt: str = ""
    full_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class DocumentAnalysisResult(BaseModel):
    """Aggregated analysis across a folder of reference documents."""

    folder_path: str
    total_files_scanned: int = 0
    supported_files_count: int = 0
    documents: list[DocumentSummary] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    available_sources: list[str] = Field(default_factory=list)
    combined_topics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def build_prompt_context(self, max_chars_per_doc: int = 4000) -> str:
        """Render a compact, structured summary of all documents for LLM prompts."""
        if not self.documents:
            return "No reference documents were found in the supplied folder."

        sections: list[str] = []
        for doc in self.documents:
            meta_parts = [f"type={doc.doc_type}", f"words={doc.word_count}"]
            if doc.page_count is not None:
                meta_parts.append(f"pages={doc.page_count}")
            if doc.row_count is not None:
                meta_parts.append(f"rows={doc.row_count}")
            if doc.columns:
                meta_parts.append(f"columns={doc.columns}")
            content = doc.full_text[:max_chars_per_doc]
            if len(doc.full_text) > max_chars_per_doc:
                content += "\n...[truncated]..."
            sections.append(
                f"### Source: {doc.source_label} ({doc.relative_path})\n"
                f"Metadata: {', '.join(meta_parts)}\n"
                f"Headings: {doc.detected_headings[:10]}\n"
                f"Content:\n{content}"
            )
        return "\n\n".join(sections)


class SourceReference(BaseModel):
    """Structured link between a requirement source name and the underlying file."""

    name: str
    filename: str | None = None
    doc_type: str | None = None
    notes: str | None = None


class ClarificationQuestion(BaseModel):
    """A single clarification question when the brief or documents have ambiguity."""

    id: str
    question: str = Field(..., min_length=5)
    reason: str
    category: Literal[
        "scope",
        "data_access",
        "audience",
        "output_format",
        "restrictions",
        "escalation",
        "ambiguity",
    ] = "ambiguity"
    suggested_options: list[str] = Field(default_factory=list)
    required: bool = True
    answer: str | None = None


class ValidationIssue(BaseModel):
    """Single validation finding produced by RequirementValidator."""

    severity: Literal["error", "warning"]
    field: str
    message: str


class ValidationReport(BaseModel):
    """Result of validating extracted AgentRequirements."""

    is_valid: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[ValidationIssue] = Field(default_factory=list)


class ExtractedRequirementsDraft(BaseModel):
    """Structured schema expected directly from the requirement extraction prompt/LLM."""

    agent_name: str = Field(..., description="Concise name of the AI agent (e.g., 'HR Policy Assistant').")
    domain: str = Field(..., description="Business domain (e.g., 'Human Resources').")
    target_audience: str = Field(..., description="Primary users (e.g., 'Employees').")
    purpose: str = Field(..., description="Clear one-sentence statement of the agent's core purpose.")
    supported_topics: list[str] = Field(default_factory=list, description="Topics supported by brief + sources.")
    unsupported_topics: list[str] = Field(
        default_factory=list,
        description="Explicitly excluded topics or requests lacking backing data.",
    )
    tone: str = Field(..., description="Expected conversational tone (e.g., 'Professional and friendly').")
    output_format: str = Field(..., description="Expected response format (e.g., 'Conversational answers').")
    restrictions: list[str] = Field(
        default_factory=list,
        description="Strict guardrails and policy boundaries (e.g., 'No invented HR policy').",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Human-readable names of authoritative sources (e.g., 'HR handbook', 'leave policy').",
    )


class ClarificationBatch(BaseModel):
    """Structured batch of clarification questions returned by an LLM (max 5)."""

    questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=MAX_CLARIFICATION_QUESTIONS)


class AgentRequirements(BaseModel):
    """Final validated requirements schema serialized to requirements.json."""

    agent_name: str
    domain: str
    target_audience: str
    purpose: str
    supported_topics: list[str] = Field(default_factory=list)
    unsupported_topics: list[str] = Field(default_factory=list)
    tone: str
    output_format: str
    restrictions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_details: list[SourceReference] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        max_length=MAX_CLARIFICATION_QUESTIONS,
    )
    clarification_answers: dict[str, str] = Field(default_factory=dict)
    needs_clarification: bool = False
    validation: ValidationReport = Field(
        default_factory=lambda: ValidationReport(is_valid=True, completeness_score=1.0, issues=[])
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {"generated_at": datetime.now(timezone.utc).isoformat()}
    )

    @model_validator(mode="after")
    def _enforce_clarification_limit(self) -> AgentRequirements:
        if len(self.clarification_questions) > MAX_CLARIFICATION_QUESTIONS:
            self.clarification_questions = self.clarification_questions[:MAX_CLARIFICATION_QUESTIONS]
        return self

