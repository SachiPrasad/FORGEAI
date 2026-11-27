"""Feature 8 — Critic & Optimizer Loop."""

from __future__ import annotations

from forgeai.optimizer.critic import FailureAnalyzer
from forgeai.optimizer.optimizer import CriticOptimizerLoop
from forgeai.optimizer.schemas import (
    MAX_OPTIMIZATION_ITERATIONS,
    FailureDiagnosis,
    FailureType,
    IterationRecord,
    OptimizationLog,
)

__all__ = [
    "MAX_OPTIMIZATION_ITERATIONS",
    "CriticOptimizerLoop",
    "FailureAnalyzer",
    "FailureDiagnosis",
    "FailureType",
    "IterationRecord",
    "OptimizationLog",
]

