"""Failure analyzer & classifier for Feature 8 — Critic & Optimizer Loop."""

from __future__ import annotations

from forgeai.eval.schemas import CaseEvaluationResult, EvalCase, EvalDataset, EvaluationReport
from forgeai.generator.schemas import AgentDefinition
from forgeai.optimizer.schemas import FailureDiagnosis, FailureType


class FailureAnalyzer:
    """Inspect failed evaluation cases and classify root causes into the 6 standard failure types."""

    def analyze(
        self,
        report: EvaluationReport,
        dataset: EvalDataset,
        agent_def: AgentDefinition,
    ) -> list[FailureDiagnosis]:
        case_map: dict[str, EvalCase] = {c.test_id: c for c in dataset.cases}
        tool_names = {t.name for t in agent_def.tools if t.enabled}
        diagnoses: list[FailureDiagnosis] = []

        for result in report.results:
            if result.passed:
                continue
            case = case_map.get(result.test_id)
            diagnoses.append(self._classify_case(result, case, tool_names))
        return diagnoses

    @staticmethod
    def _classify_case(
        result: CaseEvaluationResult,
        case: EvalCase | None,
        tool_names: set[str],
    ) -> FailureDiagnosis:
        # 1. Schema / JSON contract violation
        if not result.valid_json_contract:
            return FailureDiagnosis(
                test_id=result.test_id,
                category=result.category,
                failure_type="output_failure",
                root_cause="Agent response violated the required JSON output contract.",
                recommended_fix="Enforce strict output_contract formatting instructions in system_prompt.",
            )

        # 2. Missing essential tools
        if not ({"document_retriever", "retrieve_knowledge"} & tool_names) and not result.should_refuse:
            return FailureDiagnosis(
                test_id=result.test_id,
                category=result.category,
                failure_type="tool_failure",
                root_cause="Retrieval tool is missing or disabled on the generated agent.",
                recommended_fix="Attach and enable 'document_retriever' from the ToolRegistry.",
            )

        # 3. Refusal errors
        if result.refusal == "No":
            if result.should_refuse:
                return FailureDiagnosis(
                    test_id=result.test_id,
                    category=result.category,
                    failure_type="hallucination",
                    root_cause="Agent attempted to answer an unsupported out-of-scope question instead of refusing.",
                    recommended_fix="Strengthen unsupported-topic refusal guardrails in system_prompt.",
                )
            return FailureDiagnosis(
                test_id=result.test_id,
                category=result.category,
                failure_type="wrong_refusal",
                root_cause="Agent falsely refused an answerable question covered by available documents.",
                recommended_fix="Refine scope instructions and increase retrieval top_k for borderline policy queries.",
            )

        # 4. Retrieval miss check before groundedness (missing docs cause ungrounded answers)
        expected_sources = case.expected_sources if case else []
        retrieved_blob = " ".join(result.response.retrieved_context).lower()
        missing_in_retrieval = any(src.lower() not in retrieved_blob for src in expected_sources)

        if (
            not result.should_refuse
            and (missing_in_retrieval or len(result.response.retrieved_context) == 0)
        ):
            return FailureDiagnosis(
                test_id=result.test_id,
                category=result.category,
                failure_type="retrieval_miss",
                root_cause=(
                    f"Required source document(s) {expected_sources} were not retrieved within top_k."
                ),
                recommended_fix="Increase retrieval_config.top_k and hybrid_keyword_weight.",
            )

        # 5. Groundedness / Hallucination (context was retrieved, yet answer invented unsupported facts)
        if result.grounded == "No":
            return FailureDiagnosis(
                test_id=result.test_id,
                category=result.category,
                failure_type="hallucination",
                root_cause="Agent generated facts or numbers unsupported by retrieved document context.",
                recommended_fix="Lower model temperature and enforce strict verbatim grounding restrictions.",
            )

        # 6. Context was retrieved, but agent missed facts or citations -> Prompt defect
        return FailureDiagnosis(
            test_id=result.test_id,
            category=result.category,
            failure_type="prompt_defect",
            root_cause="Relevant chunks were retrieved, but answer omitted expected facts or multi-document details.",
            recommended_fix="Update system_prompt to require complete multi-document synthesis and explicit citations.",
        )

