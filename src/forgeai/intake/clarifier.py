"""Clarification question generator (strictly capped at 5 questions maximum)."""

from __future__ import annotations

from forgeai.intake.llm_client import complete_structured
from forgeai.intake.prompts import render_clarification_prompt
from forgeai.intake.schemas import (
    MAX_CLARIFICATION_QUESTIONS,
    ClarificationBatch,
    ClarificationQuestion,
    DocumentAnalysisResult,
    ExtractedRequirementsDraft,
    ParsedBrief,
    UserConfig,
)


class ClarificationGenerator:
    """Generate up to 5 focused clarification questions when requirements have gaps or ambiguity."""

    def generate(
        self,
        parsed_brief: ParsedBrief,
        doc_analysis: DocumentAnalysisResult,
        draft: ExtractedRequirementsDraft,
        user_config: UserConfig,
        use_llm: bool = False,
    ) -> list[ClarificationQuestion]:
        max_q = min(user_config.max_clarification_questions, MAX_CLARIFICATION_QUESTIONS)

        if use_llm:
            prompt = render_clarification_prompt(
                parsed_brief=parsed_brief,
                doc_analysis=doc_analysis,
                draft_json=draft.model_dump_json(indent=2),
                max_questions=max_q,
            )
            batch = complete_structured(
                prompt=prompt,
                response_model=ClarificationBatch,
                model=user_config.model,
                temperature=user_config.temperature,
            )
            return self._normalize_questions(batch.questions, max_q)

        return self._generate_heuristic(parsed_brief, doc_analysis, draft, user_config, max_q)

    def _generate_heuristic(
        self,
        parsed_brief: ParsedBrief,
        doc_analysis: DocumentAnalysisResult,
        draft: ExtractedRequirementsDraft,
        user_config: UserConfig,
        max_q: int,
    ) -> list[ClarificationQuestion]:
        questions: list[ClarificationQuestion] = []
        signals = set(parsed_brief.ambiguity_signals)
        lower_brief = parsed_brief.normalized_text.lower()
        has_payroll_or_balance_data = any(
            any(tok in (doc.full_text.lower() + " " + " ".join(doc.columns).lower()) for tok in ("salary", "payroll", "employee_id", "leave_balance"))
            for doc in doc_analysis.documents
        )

        # 1. Policy info vs personal calculation/balance ambiguity (e.g., HR leave assistant)
        if "unclear_policy_only_vs_balance_calculation" in signals or (
            any(w in lower_brief for w in ("leave", "pto", "attendance", "benefits"))
            and not has_payroll_or_balance_data
        ):
            questions.append(
                ClarificationQuestion(
                    id="q1",
                    question=(
                        "Should this assistant provide policy information only, or should it also "
                        "calculate an employee's leave balance?"
                    ),
                    reason=(
                        "The brief covers leave/attendance policies, but the document folder does not "
                        "contain employee-level leave balance or payroll records."
                    ),
                    category="scope",
                    suggested_options=[
                        "Policy information only (no personal calculations)",
                        "Also calculate personal leave balances (requires HRIS/payroll integration)",
                    ],
                )
            )

        # 2. Transactional permissions vs read-only guidance
        if "unclear_action_permissions_and_approval" in signals:
            questions.append(
                ClarificationQuestion(
                    id=f"q{len(questions) + 1}",
                    question=(
                        "Should the assistant only explain rules, or is it permitted to submit or "
                        "approve requests on behalf of users?"
                    ),
                    reason="Action verbs in the brief suggest possible workflow execution beyond Q&A.",
                    category="data_access",
                    suggested_options=["Read-only Q&A", "Submit draft requests with human approval"],
                )
            )

        # 3. Missing escalation / out-of-scope handling
        if "missing_escalation_or_fallback_rule" in signals:
            questions.append(
                ClarificationQuestion(
                    id=f"q{len(questions) + 1}",
                    question=(
                        "When a question is outside the uploaded documents (such as payroll disputes "
                        "or individual exceptions), how should the assistant respond or escalate?"
                    ),
                    reason="The brief does not specify a fallback or human escalation contact.",
                    category="escalation",
                    suggested_options=[
                        "Decline to answer and refer to HR support",
                        "Provide general guidance and direct to a ticketing portal",
                    ],
                )
            )

        # 4. Target audience / regional or role variations
        if "missing_target_audience" in signals or not user_config.target_audience_override:
            if not parsed_brief.audience_hints:
                questions.append(
                    ClarificationQuestion(
                        id=f"q{len(questions) + 1}",
                        question="Who is the primary target audience (e.g., all employees, managers only, or HR staff)?",
                        reason="The brief does not explicitly state the target user group.",
                        category="audience",
                        suggested_options=["All employees", "Managers", "HR staff"],
                    )
                )
            elif "human resources" in draft.domain.lower():
                questions.append(
                    ClarificationQuestion(
                        id=f"q{len(questions) + 1}",
                        question=(
                            "Do policy rules vary by employee type or region (e.g., full-time vs. "
                            "contractors, or different countries/offices)?"
                        ),
                        reason="Employee benefits and leave eligibility frequently depend on employment type or location.",
                        category="audience",
                        suggested_options=["Single standard policy for all employees", "Ask user for role/location first"],
                    )
                )

        # 5. Source citations & output format expectations
        if "unspecified_source_citation_requirement" in signals:
            questions.append(
                ClarificationQuestion(
                    id=f"q{len(questions) + 1}",
                    question=(
                        "Should responses cite the exact policy document name or section "
                        "(e.g., 'HR handbook', 'leave policy') for every answer?"
                    ),
                    reason="The brief does not specify whether source citations are mandatory.",
                    category="output_format",
                    suggested_options=["Always cite document name and section", "Cite sources only when asked"],
                )
            )

        # 6. Tone / output format fallback if still under cap and unspecified
        if len(questions) < max_q and "unspecified_tone" in signals and not user_config.tone_override:
            questions.append(
                ClarificationQuestion(
                    id=f"q{len(questions) + 1}",
                    question=(
                        "What tone and answer style do you prefer (e.g., concise bullet points vs. "
                        "warm conversational explanations)?"
                    ),
                    reason="Neither the brief nor user config specified tone or formatting preferences.",
                    category="output_format",
                    suggested_options=["Professional and friendly conversational answers", "Concise bulleted policy summaries"],
                )
            )

        return self._normalize_questions(questions, max_q)

    @staticmethod
    def _normalize_questions(
        questions: list[ClarificationQuestion],
        max_q: int,
    ) -> list[ClarificationQuestion]:
        capped = questions[: min(max_q, MAX_CLARIFICATION_QUESTIONS)]
        normalized: list[ClarificationQuestion] = []
        seen_texts: set[str] = set()
        for idx, q in enumerate(capped, start=1):
            text_key = q.question.strip().lower()
            if not text_key or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            normalized.append(q.model_copy(update={"id": q.id or f"q{idx}"}))
        return normalized[:MAX_CLARIFICATION_QUESTIONS]

