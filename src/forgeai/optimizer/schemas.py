"""Pydantic v2 schemas for Feature 8 — Critic & Optimizer Loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_OPTIMIZATION_ITERATIONS = 3

FailureType = Literal[
    "retrieval_miss",
    "prompt_defect",
    "hallucination",
    "wrong_refusal",
    "tool_failure",
    "output_failure",
]


class FailureDiagnosis(BaseModel):
    """Diagnosis for a single failed evaluation test case."""

    test_id: str
    category: str
    failure_type: FailureType
    root_cause: str
    recommended_fix: str


class IterationRecord(BaseModel):
    """Structured record of one optimization iteration in iteration_log.json."""

    iteration: int = Field(..., ge=1, le=MAX_OPTIMIZATION_ITERATIONS)
    failure_type: FailureType
    reason: str
    change_applied: str
    tests_addressed: list[str] = Field(default_factory=list)
    score_before: float = Field(..., ge=0.0, le=1.0)
    score_after: float = Field(..., ge=0.0, le=1.0)
    passed_before: int = 0
    passed_after: int = 0
    newly_passed_tests: list[str] = Field(default_factory=list)
    regressed_tests: list[str] = Field(default_factory=list)
    rolled_back: bool = False


class OptimizationLog(BaseModel):
    """Full optimization trace exported to iteration_log.json."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    max_iterations: int = Field(default=MAX_OPTIMIZATION_ITERATIONS, ge=1, le=MAX_OPTIMIZATION_ITERATIONS)
    total_iterations_run: int = 0
    initial_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    initial_passed: int = 0
    final_passed: int = 0
    total_cases: int = 0
    stop_reason: str
    iterations: list[IterationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _cap_iterations(cls, value: int) -> int:
        if isinstance(value, int) and value > MAX_OPTIMIZATION_ITERATIONS:
            return MAX_OPTIMIZATION_ITERATIONS
        return value

