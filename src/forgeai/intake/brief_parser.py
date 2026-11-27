"""Business brief parser for extracting intents, entities, constraints, and ambiguity signals."""

from __future__ import annotations

import re
from pathlib import Path

from forgeai.intake.schemas import ParsedBrief

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Human Resources": (
        "hr",
        "human resources",
        "employee",
        "employees",
        "leave",
        "pto",
        "attendance",
        "benefits",
        "onboarding",
        "payroll",
        "company rules",
        "handbook",
    ),
    "Procurement & Maritime Supply": (
        "procurement",
        "spare parts",
        "ship-spares",
        "ship spares",
        "marine",
        "vessel",
        "supplier",
        "suppliers",
        "vendor",
        "catalog",
        "inventory",
        "part number",
        "part numbers",
        "specifications",
        "rfq",
        "purchase order",
    ),
    "Customer Support": ("support", "ticket", "customer", "refund", "troubleshooting", "helpdesk"),
    "Finance & Accounting": ("invoice", "expense", "reimbursement", "budget", "tax", "accounting"),
    "Legal & Compliance": ("compliance", "contract", "gdpr", "legal", "privacy", "audit"),
    "IT & Engineering": ("it", "vpn", "laptop", "password", "devops", "incident", "software"),
    "Sales & Operations": ("sales", "crm", "lead", "pipeline", "operations", "vendor"),
}

_AUDIENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfor\s+(?:all\s+)?(employees|staff|workers|team members)\b", re.I), "Employees"),
    (re.compile(r"\bfor\s+(?:the\s+)?(procurement\s+(?:team|officers?)|buyers|fleet\s+engineers?|marine\s+engineers?|purchasing\s+team)\b", re.I), "Procurement officers and fleet engineers"),
    (re.compile(r"\bfor\s+(?:new\s+)?(hires|newcomers|onboarding employees)\b", re.I), "New hires"),
    (re.compile(r"\bfor\s+(managers|supervisors|team leads)\b", re.I), "Managers"),
    (re.compile(r"\bfor\s+(hr\s+teams?|recruiters)\b", re.I), "HR staff"),
    (re.compile(r"\bfor\s+(customers|clients|end users|users)\b", re.I), "Customers"),
    (re.compile(r"\bfor\s+(developers|engineers)\b", re.I), "Engineers"),
]

_TONE_KEYWORDS: dict[str, str] = {
    "friendly": "Professional and friendly",
    "empathetic": "Empathetic and supportive",
    "formal": "Formal and authoritative",
    "concise": "Concise and direct",
    "professional": "Professional",
    "casual": "Conversational and approachable",
}

_OUTPUT_FORMAT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjson\b", re.I), "Structured JSON"),
    (re.compile(r"\bmarkdown\b", re.I), "Markdown"),
    (re.compile(r"\bbullet(?:ed)?\b|\bchecklist\b", re.I), "Structured bullet points"),
    (re.compile(r"\banswer\s+questions\b|\bchat\b|\bconversational\b|\bq&a\b", re.I), "Conversational answers"),
]

_STOP_TOPICS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "about",
    "questions",
    "question",
    "assistant",
    "agent",
    "bot",
    "system",
    "users",
    "user",
    "employees",
    "company",
    "that",
    "which",
    "can",
    "answer",
    "help",
    "to",
    "from",
    "search",
    "find",
    "compare",
    "calculate",
    "retrieve",
    "buyers",
    "fleet",
    "engineers",
    "officers",
}


class BriefParser:
    """Parse a raw business brief string or text file into structured signals."""

    def parse(self, brief_or_path: str | Path) -> ParsedBrief:
        raw_text = self._load_brief_text(brief_or_path)
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        if not normalized:
            raise ValueError("Business brief cannot be empty.")

        lower = normalized.lower()
        words = re.findall(r"\b[\w'-]+\b", normalized)

        domain_hints = self._detect_domains(lower)
        audience_hints = self._detect_audiences(normalized)
        action_intents = self._detect_action_intents(lower)
        candidate_topics = self._extract_topics(normalized)
        requested_capabilities = self._extract_capabilities(lower)
        tone_hints = [label for kw, label in _TONE_KEYWORDS.items() if re.search(rf"\b{kw}\b", lower)]
        output_format_hints = [label for pat, label in _OUTPUT_FORMAT_HINTS if pat.search(normalized)]
        constraint_hints = self._extract_constraints(normalized)
        ambiguity_signals = self._detect_ambiguity_signals(
            normalized=normalized,
            lower=lower,
            audience_hints=audience_hints,
            tone_hints=tone_hints,
            output_format_hints=output_format_hints,
            requested_capabilities=requested_capabilities,
        )

        return ParsedBrief(
            raw_text=raw_text,
            normalized_text=normalized,
            word_count=len(words),
            domain_hints=domain_hints,
            audience_hints=audience_hints,
            action_intents=action_intents,
            candidate_topics=candidate_topics,
            requested_capabilities=requested_capabilities,
            tone_hints=tone_hints,
            output_format_hints=output_format_hints,
            constraint_hints=constraint_hints,
            ambiguity_signals=ambiguity_signals,
        )

    @staticmethod
    def _load_brief_text(brief_or_path: str | Path) -> str:
        if isinstance(brief_or_path, Path):
            return brief_or_path.read_text(encoding="utf-8")
        candidate_path = Path(brief_or_path)
        if "\n" not in brief_or_path and len(brief_or_path) < 260 and candidate_path.is_file():
            return candidate_path.read_text(encoding="utf-8")
        return brief_or_path

    @staticmethod
    def _detect_domains(lower: str) -> list[str]:
        scored: list[tuple[int, str]] = []
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lower))
            if hits > 0:
                scored.append((hits, domain))
        scored.sort(reverse=True)
        return [domain for _, domain in scored]

    @staticmethod
    def _detect_audiences(text: str) -> list[str]:
        found: list[str] = []
        for pattern, label in _AUDIENCE_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    @staticmethod
    def _detect_action_intents(lower: str) -> list[str]:
        verbs = (
            "answer",
            "explain",
            "summarize",
            "calculate",
            "compute",
            "approve",
            "submit",
            "check",
            "lookup",
            "recommend",
            "escalate",
            "generate",
        )
        return [v for v in verbs if re.search(rf"\b{v}\b", lower)]

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        topics: list[str] = []
        # Strip trailing "from <source>" clauses so source names aren't mistaken for unsupported topics
        cleaned_text = re.sub(r"\bfrom\s+(?:the\s+)?[^.?!;]+$", "", text.strip(), flags=re.IGNORECASE).strip()
        about_match = (
            re.search(r"\b(?:about|covering|regarding)\s+([^.?!;]+)", cleaned_text, flags=re.IGNORECASE)
            or re.search(r"\bto\s+([^.?!;]+)", cleaned_text, flags=re.IGNORECASE)
            or re.search(r"\b(?:on|for)\s+([^.?!;]+)", cleaned_text, flags=re.IGNORECASE)
        )
        if about_match:
            chunk = about_match.group(1)
            raw_parts = re.split(r",|\band\b|\bor\b|/", chunk, flags=re.IGNORECASE)
            for part in raw_parts:
                cleaned = part.strip(" .,:;\"'")
                if not cleaned:
                    continue
                tokens = [t for t in cleaned.lower().split() if t not in _STOP_TOPICS]
                if tokens:
                    label = " ".join(tokens).strip()
                    if label and label not in topics:
                        topics.append(label)
        return topics

    @staticmethod
    def _extract_capabilities(lower: str) -> list[str]:
        capabilities: list[str] = []
        if re.search(r"\b(answer|explain|q&a|policy|policies|rules)\b", lower):
            capabilities.append("policy_qa")
        if re.search(r"\b(calculate|balance|payroll|salary|compensation|deduction|prorate)\b", lower):
            capabilities.append("personal_or_numeric_calculation")
        if re.search(r"\b(apply|submit|approve|request|book|update|modify)\b", lower):
            capabilities.append("transactional_action")
        return capabilities

    @staticmethod
    def _extract_constraints(text: str) -> list[str]:
        constraints: list[str] = []
        for sentence in re.split(r"[.?!]\s*", text):
            s = sentence.strip()
            if re.search(r"\b(must not|never|only|do not|don't|no\s+\w+|cannot|restricted)\b", s, re.I):
                constraints.append(s)
        return constraints

    @staticmethod
    def _detect_ambiguity_signals(
        normalized: str,
        lower: str,
        audience_hints: list[str],
        tone_hints: list[str],
        output_format_hints: list[str],
        requested_capabilities: list[str],
    ) -> list[str]:
        signals: list[str] = []
        if len(normalized.split()) < 15:
            signals.append("brief_is_very_short")
        if not audience_hints:
            signals.append("missing_target_audience")
        if not tone_hints:
            signals.append("unspecified_tone")
        if not output_format_hints:
            signals.append("unspecified_output_format")
        if "policy_qa" in requested_capabilities and "personal_or_numeric_calculation" not in requested_capabilities:
            if any(term in lower for term in ("leave", "pto", "attendance", "benefits", "allowance")):
                signals.append("unclear_policy_only_vs_balance_calculation")
        if "transactional_action" in requested_capabilities:
            signals.append("unclear_action_permissions_and_approval")
        if not re.search(r"\b(escalat|human|hr\s+team|fallback|don't know|unknown|outside)\b", lower):
            signals.append("missing_escalation_or_fallback_rule")
        if not re.search(r"\b(cite|citation|source|reference|link)\b", lower):
            signals.append("unspecified_source_citation_requirement")
        return signals

