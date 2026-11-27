"""Pydantic v2 schemas for Feature 6 (evals.json) and Feature 7 (Evaluation Judge)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

EvalCategory = Literal["answerable", "multi_document", "edge_case", "ambiguous", "unsupported"]
TriState = Literal["Yes", "No", "N/A"]

MIN_EVAL_CASES = 20
DEFAULT_CATEGORY_COUNTS: dict[EvalCategory, int] = {
    "answerable": 8,
    "multi_document": 4,
    "edge_case": 3,
    "ambiguous": 2,
    "unsupported": 3,
}


class EvalCase(BaseModel):
    """Single evaluation test case in evals.json."""

    test_id: str = Field(..., min_length=1)
    category: EvalCategory
    question: str = Field(..., min_length=5)
    expected_answer: str = ""
    expected_sources: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    source_chunk_ids: list[str] = Field(default_factory=list)


class EvalCaseBatch(BaseModel):
    """LLM-generated batch of evaluation cases."""

    cases: list[EvalCase] = Field(default_factory=list)


class EvalDataset(BaseModel):
    """Validated evaluation dataset exported to evals.json (minimum 20 unique cases)."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    total_cases: int = Field(..., ge=MIN_EVAL_CASES)
    category_counts: dict[str, int] = Field(default_factory=dict)
    cases: list[EvalCase] = Field(..., min_length=MIN_EVAL_CASES)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts_and_uniqueness(self) -> EvalDataset:
        if len(self.cases) < MIN_EVAL_CASES:
            raise ValueError(f"evals.json must contain at least {MIN_EVAL_CASES} test cases.")
        seen_ids: set[str] = set()
        for case in self.cases:
            if case.test_id in seen_ids:
                raise ValueError(f"Duplicate test_id found: {case.test_id}")
            seen_ids.add(case.test_id)
        self.total_cases = len(self.cases)
        return self


class AgentResponse(BaseModel):
    """Structured response produced by executing the generated agent against a question."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "unsupported"] = "medium"
    refused_unsupported_topic: bool = False
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
    raw_json: dict[str, Any] = Field(default_factory=dict)


class LLMJudgeVerdict(BaseModel):
    """Structured semantic verdict returned by the optional LLM judge."""

    grounded: bool
    correct: bool
    refusal_appropriate: bool
    reasoning: str


class CaseEvaluationResult(BaseModel):
    """Per-test-case hybrid evaluation result."""

    test_id: str
    category: EvalCategory
    question: str
    should_refuse: bool
    response: AgentResponse
    grounded: TriState
    correct: TriState
    refusal: TriState
    source_quality: TriState
    valid_json_contract: bool
    passed: bool
    score_label: Literal["Pass", "Fail"]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    deterministic_checks: dict[str, bool] = Field(default_factory=dict)
    llm_judge_notes: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)


class EvaluationScorecard(BaseModel):
    """Aggregated evaluation metrics across all test cases."""

    agent_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    groundedness_rate: float = Field(..., ge=0.0, le=1.0)
    correctness_rate: float = Field(..., ge=0.0, le=1.0)
    refusal_accuracy_rate: float = Field(..., ge=0.0, le=1.0)
    source_quality_rate: float = Field(..., ge=0.0, le=1.0)
    contract_compliance_rate: float = Field(..., ge=0.0, le=1.0)
    category_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Complete evaluation report (scorecard + individual test results)."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluation_mode: str
    scorecard: EvaluationScorecard
    results: list[CaseEvaluationResult] = Field(default_factory=list)
    formatted_scorecard: str = ""

