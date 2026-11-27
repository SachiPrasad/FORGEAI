"""Main orchestrator for Feature 1 — Intelligent Intake Interviewer."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forgeai.intake.brief_parser import BriefParser
from forgeai.intake.clarifier import ClarificationGenerator
from forgeai.intake.document_analyzer import DocumentAnalyzer
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials
from forgeai.intake.prompts import render_extraction_prompt, render_refinement_prompt
from forgeai.intake.schemas import (
    AgentRequirements,
    DocumentAnalysisResult,
    ExtractedRequirementsDraft,
    ParsedBrief,
    SourceReference,
    UserConfig,
)
from forgeai.intake.validator import RequirementValidator


class IntakeInterviewer:
    """Analyze a business brief, document folder, and user config into validated requirements.json."""

    def __init__(
        self,
        brief_parser: BriefParser | None = None,
        document_analyzer: DocumentAnalyzer | None = None,
        clarifier: ClarificationGenerator | None = None,
        validator: RequirementValidator | None = None,
    ) -> None:
        self.brief_parser = brief_parser or BriefParser()
        self.document_analyzer = document_analyzer or DocumentAnalyzer()
        self.clarifier = clarifier or ClarificationGenerator()
        self.validator = validator or RequirementValidator()

    def run(
        self,
        brief: str | Path,
        docs_dir: str | Path,
        config: UserConfig | dict[str, Any] | str | Path | None = None,
        output_path: str | Path | None = None,
        clarification_answers: dict[str, str] | None = None,
    ) -> AgentRequirements:
        """Execute the full intake pipeline and optionally persist requirements.json."""
        user_config = self._load_user_config(config)
        parsed_brief = self.brief_parser.parse(brief)
        doc_analysis = self.document_analyzer.analyze_folder(
            docs_dir,
            include_subdirectories=user_config.include_subdirectories,
        )

        use_llm = self._should_use_llm(user_config)
        draft = self._extract_draft(parsed_brief, doc_analysis, user_config, use_llm=use_llm)
        draft = self._apply_user_overrides(draft, user_config)

        questions = self.clarifier.generate(
            parsed_brief=parsed_brief,
            doc_analysis=doc_analysis,
            draft=draft,
            user_config=user_config,
            use_llm=use_llm,
        )

        source_details = [
            SourceReference(
                name=doc.source_label,
                filename=doc.relative_path,
                doc_type=doc.doc_type,
                notes=f"{doc.word_count} words" + (f", {doc.row_count} rows" if doc.row_count is not None else ""),
            )
            for doc in doc_analysis.documents
        ]

        requirements = AgentRequirements(
            agent_name=draft.agent_name,
            domain=draft.domain,
            target_audience=draft.target_audience,
            purpose=draft.purpose,
            supported_topics=draft.supported_topics,
            unsupported_topics=draft.unsupported_topics,
            tone=draft.tone,
            output_format=draft.output_format,
            restrictions=draft.restrictions,
            sources=draft.sources,
            source_details=source_details,
            clarification_questions=questions,
            needs_clarification=len(questions) > 0,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "extraction_mode": "llm" if use_llm else "heuristic",
                "model": user_config.model if use_llm else None,
                "brief_word_count": parsed_brief.word_count,
                "documents_scanned": doc_analysis.supported_files_count,
                "skipped_files": doc_analysis.skipped_files,
                "document_warnings": doc_analysis.warnings,
            },
        )

        if clarification_answers:
            requirements = self.refine_with_answers(
                requirements=requirements,
                answers=clarification_answers,
                user_config=user_config,
                doc_analysis=doc_analysis,
            )
        else:
            requirements.validation = self.validator.validate(requirements, doc_analysis=doc_analysis)

        if output_path is not None:
            self.save_requirements(requirements, output_path)

        return requirements

    def refine_with_answers(
        self,
        requirements: AgentRequirements,
        answers: dict[str, str],
        user_config: UserConfig | dict[str, Any] | None = None,
        doc_analysis: DocumentAnalysisResult | None = None,
        output_path: str | Path | None = None,
    ) -> AgentRequirements:
        """Incorporate user answers to clarification questions and re-validate."""
        cfg = self._load_user_config(user_config)
        use_llm = self._should_use_llm(cfg)

        qa_pairs: list[dict[str, str]] = []
        updated_questions = []
        merged_answers = dict(requirements.clarification_answers)
        for q in requirements.clarification_questions:
            ans = answers.get(q.id) or answers.get(q.question) or q.answer
            if ans:
                ans = ans.strip()
                merged_answers[q.id] = ans
                qa_pairs.append({"id": q.id, "question": q.question, "answer": ans})
                updated_questions.append(q.model_copy(update={"answer": ans}))
            else:
                updated_questions.append(q)

        if qa_pairs:
            if use_llm:
                prompt = render_refinement_prompt(requirements, qa_pairs)
                refined_draft = complete_structured(
                    prompt=prompt,
                    response_model=ExtractedRequirementsDraft,
                    model=cfg.model,
                    temperature=cfg.temperature,
                )
            else:
                refined_draft = self._refine_heuristically(requirements, qa_pairs)

            refined_draft = self._apply_user_overrides(refined_draft, cfg)
            requirements = requirements.model_copy(
                update={
                    "agent_name": refined_draft.agent_name,
                    "domain": refined_draft.domain,
                    "target_audience": refined_draft.target_audience,
                    "purpose": refined_draft.purpose,
                    "supported_topics": refined_draft.supported_topics,
                    "unsupported_topics": refined_draft.unsupported_topics,
                    "tone": refined_draft.tone,
                    "output_format": refined_draft.output_format,
                    "restrictions": refined_draft.restrictions,
                    "sources": refined_draft.sources,
                    "clarification_questions": updated_questions,
                    "clarification_answers": merged_answers,
                    "needs_clarification": any(q.required and not q.answer for q in updated_questions),
                }
            )
        else:
            requirements = requirements.model_copy(
                update={
                    "clarification_questions": updated_questions,
                    "clarification_answers": merged_answers,
                }
            )

        requirements.validation = self.validator.validate(requirements, doc_analysis=doc_analysis)

        if output_path is not None:
            self.save_requirements(requirements, output_path)

        return requirements

    @staticmethod
    def save_requirements(requirements: AgentRequirements, output_path: str | Path) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(requirements.model_dump_json(indent=2), encoding="utf-8")
        return dest

    @staticmethod
    def _load_user_config(config: UserConfig | dict[str, Any] | str | Path | None) -> UserConfig:
        if config is None:
            return UserConfig()
        if isinstance(config, UserConfig):
            return config
        if isinstance(config, (str, Path)):
            path = Path(config)
            data = json.loads(path.read_text(encoding="utf-8"))
            return UserConfig.model_validate(data)
        return UserConfig.model_validate(config)

    @staticmethod
    def _should_use_llm(user_config: UserConfig) -> bool:
        if user_config.use_llm is not None:
            return user_config.use_llm
        return has_configured_llm_credentials(user_config.model)

    def _extract_draft(
        self,
        parsed_brief: ParsedBrief,
        doc_analysis: DocumentAnalysisResult,
        user_config: UserConfig,
        use_llm: bool,
    ) -> ExtractedRequirementsDraft:
        if use_llm:
            prompt = render_extraction_prompt(parsed_brief, doc_analysis, user_config)
            return complete_structured(
                prompt=prompt,
                response_model=ExtractedRequirementsDraft,
                model=user_config.model,
                temperature=user_config.temperature,
            )
        return self._extract_heuristically(parsed_brief, doc_analysis)

    def _extract_heuristically(
        self,
        parsed_brief: ParsedBrief,
        doc_analysis: DocumentAnalysisResult,
    ) -> ExtractedRequirementsDraft:
        all_doc_text = " ".join(doc.full_text.lower() for doc in doc_analysis.documents)
        all_columns = " ".join(col.lower() for doc in doc_analysis.documents for col in doc.columns)

        domain = parsed_brief.domain_hints[0] if parsed_brief.domain_hints else "General Business"
        if domain == "General Business" and any(
            kw in all_doc_text for kw in ("leave", "employee", "hr ", "benefits", "attendance")
        ):
            domain = "Human Resources"

        target_audience = parsed_brief.audience_hints[0] if parsed_brief.audience_hints else "End users"
        agent_name = self._derive_agent_name(domain, parsed_brief)
        purpose = self._derive_purpose(domain, parsed_brief)

        # Supported topics from brief topics + matched doc headings/topics
        supported_topics: list[str] = []
        for topic in parsed_brief.candidate_topics:
            formatted = self._format_topic_label(topic)
            if formatted not in supported_topics:
                supported_topics.append(formatted)

        for doc in doc_analysis.documents:
            column_set = {c.lower() for c in doc.columns}
            for heading in doc.detected_headings:
                clean = heading.strip()
                if (
                    clean
                    and clean.lower() not in column_set
                    and "_" not in clean
                    and not clean.lower().startswith("[page ")
                    and len(clean.split()) <= 4
                ):
                    # Only promote headings relevant to brief or core policy/domain sections
                    if any(
                        tok in clean.lower()
                        for tok in (
                            "leave",
                            "benefit",
                            "attendance",
                            "rule",
                            "policy",
                            "holiday",
                            "pto",
                            "inventory",
                            "spare",
                            "supplier",
                            "specification",
                            "procurement",
                            "catalog",
                        )
                    ):
                        formatted = self._format_topic_label(clean)
                        if formatted not in supported_topics:
                            supported_topics.append(formatted)

        if not supported_topics:
            supported_topics = [self._format_topic_label(t) for t in doc_analysis.combined_topics[:5]] or ["General Q&A"]

        # Unsupported topics grounded in missing data
        unsupported_topics: list[str] = []
        if domain == "Human Resources":
            has_payroll_data = any(
                kw in (all_doc_text + " " + all_columns)
                for kw in ("salary_amount", "payroll_id", "net_pay", "tax_withholding", "bank_account")
            )
            if not has_payroll_data:
                unsupported_topics.append("Payroll calculations without payroll data")
            has_balance_data = any(
                kw in (all_doc_text + " " + all_columns)
                for kw in ("leave_balance", "remaining_pto", "employee_id")
            )
            if not has_balance_data:
                unsupported_topics.append("Individual employee leave balance lookups without HRIS data")
        elif "procurement" in domain.lower():
            unsupported_topics.append("Live ERP purchase order execution without ERP integration")
            unsupported_topics.append("Real-time customs clearance or competitor confidential pricing")
        elif "personal_or_numeric_calculation" not in parsed_brief.requested_capabilities and not doc_analysis.documents:
            unsupported_topics.append("Requests outside the provided reference documents")

        tone = parsed_brief.tone_hints[0] if parsed_brief.tone_hints else "Professional and friendly"
        output_format = (
            parsed_brief.output_format_hints[0]
            if parsed_brief.output_format_hints
            else "Conversational answers"
        )

        restrictions = list(parsed_brief.constraint_hints)
        if domain == "Human Resources":
            restrictions.append("No invented HR policy")
        else:
            restrictions.append(f"No invented {domain.lower()} policy or unverified claims")
        restrictions.append("Answer strictly from provided reference documents and state when information is missing")

        sources = list(doc_analysis.available_sources)

        return ExtractedRequirementsDraft(
            agent_name=agent_name,
            domain=domain,
            target_audience=target_audience,
            purpose=purpose,
            supported_topics=self._dedupe(supported_topics),
            unsupported_topics=self._dedupe(unsupported_topics),
            tone=tone,
            output_format=output_format,
            restrictions=self._dedupe(restrictions),
            sources=self._dedupe(sources),
        )

    @staticmethod
    def _apply_user_overrides(
        draft: ExtractedRequirementsDraft,
        user_config: UserConfig,
    ) -> ExtractedRequirementsDraft:
        updates: dict[str, Any] = {}
        if user_config.agent_name_override:
            updates["agent_name"] = user_config.agent_name_override
        if user_config.domain_override:
            updates["domain"] = user_config.domain_override
        if user_config.target_audience_override:
            updates["target_audience"] = user_config.target_audience_override
        if user_config.tone_override:
            updates["tone"] = user_config.tone_override
        if user_config.output_format_override:
            updates["output_format"] = user_config.output_format_override
        if user_config.extra_restrictions:
            updates["restrictions"] = IntakeInterviewer._dedupe(
                [*draft.restrictions, *user_config.extra_restrictions]
            )
        if user_config.extra_unsupported_topics:
            updates["unsupported_topics"] = IntakeInterviewer._dedupe(
                [*draft.unsupported_topics, *user_config.extra_unsupported_topics]
            )
        return draft.model_copy(update=updates) if updates else draft

    @staticmethod
    def _refine_heuristically(
        requirements: AgentRequirements,
        qa_pairs: list[dict[str, str]],
    ) -> ExtractedRequirementsDraft:
        supported = list(requirements.supported_topics)
        unsupported = list(requirements.unsupported_topics)
        restrictions = list(requirements.restrictions)
        purpose = requirements.purpose
        tone = requirements.tone
        output_format = requirements.output_format
        target_audience = requirements.target_audience

        for item in qa_pairs:
            q_lower = item["question"].lower()
            a_text = item["answer"].strip()
            a_lower = a_text.lower()

            if "leave balance" in q_lower or "policy information only" in q_lower:
                if any(tok in a_lower for tok in ("policy information only", "only policy", "policy only", "no calculation")):
                    if "Leave balance calculations without HRIS data" not in unsupported:
                        unsupported.append("Leave balance calculations without HRIS data")
                    restrictions.append("Provide policy information only; do not compute personal leave balances")
                elif any(tok in a_lower for tok in ("calculate", "both", "yes")):
                    if "Leave balance calculation" not in supported:
                        supported.append("Leave balance calculation")
                    unsupported = [u for u in unsupported if "leave balance" not in u.lower()]
                    purpose = f"{purpose.rstrip('.')} and calculate leave balances when employee data is supplied."

            if "escalate" in q_lower or "outside" in q_lower:
                restrictions.append(f"Escalation rule: {a_text}")

            if "cite" in q_lower or "source" in q_lower:
                if any(tok in a_lower for tok in ("always", "yes", "cite")):
                    restrictions.append("Always cite the exact source document name in responses")

            if "target audience" in q_lower and len(a_text.split()) <= 6:
                target_audience = a_text

            if "tone" in q_lower or "answer style" in q_lower:
                if "bullet" in a_lower:
                    output_format = "Concise bulleted summaries"
                if "friendly" in a_lower or "professional" in a_lower:
                    tone = "Professional and friendly"

            if "region" in q_lower or "employee type" in q_lower:
                restrictions.append(f"Audience scope rule: {a_text}")

        return ExtractedRequirementsDraft(
            agent_name=requirements.agent_name,
            domain=requirements.domain,
            target_audience=target_audience,
            purpose=purpose,
            supported_topics=IntakeInterviewer._dedupe(supported),
            unsupported_topics=IntakeInterviewer._dedupe(unsupported),
            tone=tone,
            output_format=output_format,
            restrictions=IntakeInterviewer._dedupe(restrictions),
            sources=list(requirements.sources),
        )

    @staticmethod
    def _derive_agent_name(domain: str, parsed_brief: ParsedBrief) -> str:
        lower = parsed_brief.normalized_text.lower()
        if domain == "Human Resources":
            if "policy" in lower or "policies" in lower or "rules" in lower:
                return "HR Policy Assistant"
            return "HR Assistant"
        if "procurement" in domain.lower():
            if "ship" in lower or "marine" in lower or "spare" in lower:
                return "Ship-Spares Procurement Assistant"
            return "Procurement Assistant"
        short_domain = domain.split("&")[0].strip()
        return f"{short_domain} Assistant"

    @staticmethod
    def _derive_purpose(domain: str, parsed_brief: ParsedBrief) -> str:
        if domain == "Human Resources":
            return "Answer HR policy questions"
        if "procurement" in domain.lower():
            return "Search spare parts inventory, compare specifications, and retrieve supplier & procurement policies"
        text = parsed_brief.normalized_text
        match = re.search(r"\bthat\s+can\s+([^.?!]+)", text, flags=re.I)
        if match:
            phrase = match.group(1).strip()
            return phrase[0].upper() + phrase[1:]
        return f"Assist {parsed_brief.audience_hints[0].lower() if parsed_brief.audience_hints else 'users'} with {domain.lower()} questions"

    @staticmethod
    def _format_topic_label(topic: str) -> str:
        cleaned = re.sub(r"\bpolicies\b|\bpolicy\b|\brules\b", "", topic, flags=re.I).strip()
        cleaned = re.sub(r"\s+", " ", cleaned) or topic.strip()
        return cleaned[:1].upper() + cleaned[1:]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            norm = item.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(item.strip())
        return result
