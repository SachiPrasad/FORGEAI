"""Structured tracing, latency measurement, and approximate token/cost tracking."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

# Approximate per-1K token USD pricing table for cost estimation across common models
_MODEL_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "openai/o3": (0.00200, 0.00800),
    "o3": (0.00200, 0.00800),
    "gemini/gemini-2.5-flash": (0.00015, 0.00060),
}


class TraceEvent(BaseModel):
    """Single traced span across pipeline stages, retrievals, tools, or LLM calls."""

    event_id: str
    stage: str
    operation: str
    status: str = "ok"
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TraceSummary(BaseModel):
    """Aggregated observability & cost report exported to trace_log.json."""

    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str
    total_events: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    events: list[TraceEvent] = Field(default_factory=list)


class TraceCollector:
    """Collect spans, latency, approximate token counts, and estimated USD cost."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.events: list[TraceEvent] = []

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_cost(self, input_tokens: int, output_tokens: int, is_llm_call: bool = True) -> float:
        if not is_llm_call:
            return 0.0
        in_rate, out_rate = _MODEL_PRICING_PER_1K.get(self.model, (0.0005, 0.0015))
        return round((input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate, 6)

    def record(
        self,
        stage: str,
        operation: str,
        duration_ms: float = 0.0,
        status: str = "ok",
        input_tokens: int = 0,
        output_tokens: int = 0,
        is_llm_call: bool = False,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TraceEvent:
        cost = self.estimate_cost(input_tokens, output_tokens, is_llm_call=is_llm_call)
        event = TraceEvent(
            event_id=f"ev_{len(self.events) + 1:03d}",
            stage=stage,
            operation=operation,
            status=status,
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            attributes=attributes or {},
            error=error,
        )
        self.events.append(event)
        return event

    @contextmanager
    def span(
        self,
        stage: str,
        operation: str,
        attributes: dict[str, Any] | None = None,
        is_llm_call: bool = False,
    ) -> Iterator[dict[str, Any]]:
        ctx: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "attributes": dict(attributes or {})}
        start = time.perf_counter()
        try:
            yield ctx
            elapsed = (time.perf_counter() - start) * 1000.0
            self.record(
                stage=stage,
                operation=operation,
                duration_ms=elapsed,
                status="ok",
                input_tokens=int(ctx.get("input_tokens", 0)),
                output_tokens=int(ctx.get("output_tokens", 0)),
                is_llm_call=is_llm_call,
                attributes=ctx.get("attributes"),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.record(
                stage=stage,
                operation=operation,
                duration_ms=elapsed,
                status="error",
                input_tokens=int(ctx.get("input_tokens", 0)),
                output_tokens=int(ctx.get("output_tokens", 0)),
                is_llm_call=is_llm_call,
                attributes=ctx.get("attributes"),
                error=str(exc),
            )
            raise

    def summary(self) -> TraceSummary:
        return TraceSummary(
            model=self.model,
            total_events=len(self.events),
            error_count=sum(1 for e in self.events if e.status == "error"),
            total_duration_ms=round(sum(e.duration_ms for e in self.events), 2),
            total_input_tokens=sum(e.input_tokens for e in self.events),
            total_output_tokens=sum(e.output_tokens for e in self.events),
            total_estimated_cost_usd=round(sum(e.estimated_cost_usd for e in self.events), 6),
            events=list(self.events),
        )

    def save_json(self, output_path: str | Path) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.summary().model_dump_json(indent=2), encoding="utf-8")
        return dest

