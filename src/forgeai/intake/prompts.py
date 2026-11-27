"""Jinja2 prompt templates for Feature 1 — Intelligent Intake Interviewer."""

from __future__ import annotations

from jinja2 import Template

from forgeai.intake.schemas import (
    MAX_CLARIFICATION_QUESTIONS,
    AgentRequirements,
    DocumentAnalysisResult,
    ParsedBrief,
    UserConfig,
)

REQUIREMENT_EXTRACTION_TEMPLATE = """You are an expert AI Systems Analyst and Requirements Architect.
Your job is to extract a precise, grounded specification for a proposed AI agent by combining:
1. The business brief
2. The analyzed reference documents
3. Optional user configuration overrides

## Business Brief
{{ brief_text }}

## Parsed Brief Signals
- Domain hints: {{ parsed_brief.domain_hints }}
- Audience hints: {{ parsed_brief.audience_hints }}
- Candidate topics: {{ parsed_brief.candidate_topics }}
- Requested capabilities: {{ parsed_brief.requested_capabilities }}
- Explicit constraints: {{ parsed_brief.constraint_hints }}
- Ambiguity signals: {{ parsed_brief.ambiguity_signals }}

## Reference Documents Summary
Available sources: {{ doc_analysis.available_sources }}
Combined topics found in documents: {{ doc_analysis.combined_topics }}

{{ document_context }}

## Optional User Configuration
{{ user_config_json }}

## Extraction Rules
1. Populate every field cleanly and concisely:
   - `agent_name`: Short descriptive name (e.g., "HR Policy Assistant").
   - `domain`: High-level domain (e.g., "Human Resources").
   - `target_audience`: Intended end users (e.g., "Employees").
   - `purpose`: Clear summary of what the agent does.
   - `supported_topics`: Only topics backed by BOTH the brief's intent AND the available reference documents (or clearly requested when no contradicting gaps exist).
   - `unsupported_topics`: Explicitly identify capabilities/topics that should be excluded because required data/systems are absent from the documents (for example, "Payroll calculations without payroll data" or "Personal leave balance lookups without HRIS integration").
   - `tone`: Appropriate communication style (honor user config override if provided).
   - `output_format`: Expected response style (e.g., "Conversational answers").
   - `restrictions`: Concrete safety & grounding rules (e.g., "No invented HR policy", "Answer only from verified company documents").
   - `sources`: Human-readable names of the actual documents used from the folder (e.g., ["HR handbook", "leave policy"]). Never invent sources not present in the folder.
2. Respect any non-null overrides in User Configuration.
3. Return valid JSON matching the ExtractedRequirementsDraft schema.
"""

CLARIFICATION_PROMPT_TEMPLATE = """You are an Intelligent Intake Interviewer conducting requirements elicitation for an AI agent.

## Business Brief
{{ brief_text }}

## Ambiguity Signals Detected
{{ parsed_brief.ambiguity_signals }}

## Available Documents
{{ doc_analysis.available_sources }}

## Current Extracted Requirements Draft
{{ draft_json }}

## Task
Generate ONLY high-value clarification questions needed to remove ambiguity before building the agent.
Rules:
- Generate AT MOST {{ max_questions }} questions (never exceed {{ max_questions }}).
- Only ask questions whose answers are NOT already clearly answered in the brief, documents, or user configuration.
- Prioritize operational scope boundaries (e.g., policy information vs. personal calculations/actions), data/system integration assumptions, target audience access rules, escalation paths, and citation/output requirements.
- Example of a great question:
  "Should this assistant provide policy information only, or should it also calculate an employee's leave balance?"
- If everything is completely clear and unambiguous, return an empty `questions` list.
- Return valid JSON matching the ClarificationBatch schema.
"""

REFINEMENT_PROMPT_TEMPLATE = """You are updating an AI agent's structured requirements specification using the stakeholder's clarification answers.

## Original Requirements
{{ requirements_json }}

## Clarification Questions & Stakeholder Answers
{% for item in qa_pairs %}
- [{{ item.id }}] Q: {{ item.question }}
  A: {{ item.answer }}
{% endfor %}

## Task
Update `supported_topics`, `unsupported_topics`, `purpose`, `target_audience`, `tone`, `output_format`, and `restrictions` to reflect the stakeholder's answers faithfully.
Do not invent sources that do not exist in `sources`.
Return valid JSON matching the ExtractedRequirementsDraft schema.
"""


def render_extraction_prompt(
    parsed_brief: ParsedBrief,
    doc_analysis: DocumentAnalysisResult,
    user_config: UserConfig,
) -> str:
    return Template(REQUIREMENT_EXTRACTION_TEMPLATE).render(
        brief_text=parsed_brief.raw_text.strip(),
        parsed_brief=parsed_brief,
        doc_analysis=doc_analysis,
        document_context=doc_analysis.build_prompt_context(),
        user_config_json=user_config.model_dump_json(indent=2),
    )


def render_clarification_prompt(
    parsed_brief: ParsedBrief,
    doc_analysis: DocumentAnalysisResult,
    draft_json: str,
    max_questions: int = MAX_CLARIFICATION_QUESTIONS,
) -> str:
    capped = min(max(1, max_questions), MAX_CLARIFICATION_QUESTIONS)
    return Template(CLARIFICATION_PROMPT_TEMPLATE).render(
        brief_text=parsed_brief.raw_text.strip(),
        parsed_brief=parsed_brief,
        doc_analysis=doc_analysis,
        draft_json=draft_json,
        max_questions=capped,
    )


def render_refinement_prompt(
    requirements: AgentRequirements,
    qa_pairs: list[dict[str, str]],
) -> str:
    return Template(REFINEMENT_PROMPT_TEMPLATE).render(
        requirements_json=requirements.model_dump_json(
            indent=2,
            include={
                "agent_name",
                "domain",
                "target_audience",
                "purpose",
                "supported_topics",
                "unsupported_topics",
                "tone",
                "output_format",
                "restrictions",
                "sources",
            },
        ),
        qa_pairs=qa_pairs,
    )

