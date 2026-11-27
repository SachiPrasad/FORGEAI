"""Safety guardrails: prompt-injection detection and tool permission verification."""

from __future__ import annotations

import re

from forgeai.generator.schemas import AgentDefinition

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(?:reveal|print|show|dump|leak)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions)\b", re.I),
    re.compile(r"\b(?:bypass|override|disable)\s+(?:all\s+)?(?:restrictions|guardrails|safety|rules)\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:dan|developer\s+mode|unrestricted)\b", re.I),
    re.compile(r"\bpretend\s+missing\s+documents?\s+exist\b", re.I),
)


def detect_prompt_injection(message: str) -> str | None:
    """Return matched pattern reason if a prompt-injection attempt is detected, else None."""
    for pat in _INJECTION_PATTERNS:
        match = pat.search(message)
        if match:
            return f"Blocked unsafe instruction override pattern: '{match.group(0)}'"
    return None


def is_tool_permitted(agent_def: AgentDefinition, tool_name: str) -> bool:
    """Verify that a tool is explicitly enabled in agent_def.tools."""
    return any(t.name == tool_name and t.enabled for t in agent_def.tools)

