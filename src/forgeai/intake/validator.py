"""Requirement validator ensuring completeness, consistency, grounding, and question caps."""

from __future__ import annotations

from forgeai.intake.schemas import (
    MAX_CLARIFICATION_QUESTIONS,
    AgentRequirements,
    DocumentAnalysisResult,
    ValidationIssue,
    ValidationReport,
)


class RequirementValidator:
    """Validate extracted AgentRequirements against schema, grounding, and assignment rules."""

    def validate(
        self,
        requirements: AgentRequirements,
        doc_analysis: DocumentAnalysisResult | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []

        required_text_fields = {
            "agent_name": requirements.agent_name,
            "domain": requirements.domain,
            "target_audience": requirements.target_audience,
            "purpose": requirements.purpose,
            "tone": requirements.tone,
            "output_format": requirements.output_format,
        }
        for field_name, value in required_text_fields.items():
            if not value or not value.strip():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field=field_name,
                        message=f"Required field '{field_name}' must not be empty.",
                    )
                )

        if not requirements.supported_topics:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="supported_topics",
                    message="At least one supported topic must be extracted.",
                )
            )

        if not requirements.restrictions:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="restrictions",
                    message="No restrictions defined; consider adding grounding guardrails (e.g., 'No invented policy').",
                )
            )

        if len(requirements.clarification_questions) > MAX_CLARIFICATION_QUESTIONS:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="clarification_questions",
                    message=(
                        f"Clarification questions ({len(requirements.clarification_questions)}) "
                        f"exceed the maximum allowed ({MAX_CLARIFICATION_QUESTIONS})."
                    ),
                )
            )

        # Check overlap between supported and unsupported topics
        supported_lower = {t.strip().lower() for t in requirements.supported_topics}
        unsupported_lower = {t.strip().lower() for t in requirements.unsupported_topics}
        overlap = supported_lower & unsupported_lower
        if overlap:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="supported_topics",
                    message=f"Topics appear in both supported and unsupported lists: {sorted(overlap)}.",
                )
            )

        # Grounding checks against analyzed documents
        if doc_analysis is not None:
            if doc_analysis.supported_files_count > 0 and not requirements.sources:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field="sources",
                        message="Documents were provided, but 'sources' is empty.",
                    )
                )
            available_lower = {s.strip().lower() for s in doc_analysis.available_sources}
            available_files_lower = {d.filename.strip().lower() for d in doc_analysis.documents}
            for src in requirements.sources:
                src_norm = src.strip().lower()
                if src_norm not in available_lower and src_norm not in available_files_lower:
                    # Allow partial substring match (e.g., "HR handbook" matching "HR handbook v2")
                    matched = any(src_norm in avail or avail in src_norm for avail in available_lower)
                    if not matched:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                field="sources",
                                message=(
                                    f"Source '{src}' does not directly match any analyzed document "
                                    f"({doc_analysis.available_sources})."
                                ),
                            )
                        )

        # Completeness score calculation
        scored_checks = [
            bool(requirements.agent_name.strip()),
            bool(requirements.domain.strip()),
            bool(requirements.target_audience.strip()),
            bool(requirements.purpose.strip()),
            bool(requirements.supported_topics),
            bool(requirements.unsupported_topics),
            bool(requirements.tone.strip()),
            bool(requirements.output_format.strip()),
            bool(requirements.restrictions),
            bool(requirements.sources) or (doc_analysis is not None and doc_analysis.supported_files_count == 0),
        ]
        completeness_score = round(sum(1 for ok in scored_checks if ok) / len(scored_checks), 2)
        has_errors = any(issue.severity == "error" for issue in issues)

        return ValidationReport(
            is_valid=not has_errors,
            completeness_score=completeness_score,
            issues=issues,
        )

