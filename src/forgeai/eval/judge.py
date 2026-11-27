"""Feature 7 — Hybrid Evaluation Judge (Deterministic Checks + Optional LLM Judge)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from forgeai.eval.agent_runner import GeneratedAgentRunner
from forgeai.eval.schemas import (
    AgentResponse,
    CaseEvaluationResult,
    EvalCase,
    EvalDataset,
    EvaluationReport,
    EvaluationScorecard,
    LLMJudgeVerdict,
    TriState,
)
from forgeai.generator.schemas import AgentDefinition
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials

_REFUSAL_PATTERNS = (
    re.compile(r"\b(?:cannot answer|can't answer|unable to answer|do not have|don't have|not covered)\b", re.I),
    re.compile(r"\b(?:outside the approved|unsupported|no supporting|unavailable in|does not contain)\b", re.I),
)


class EvaluationJudge:
    """Execute generated agent over evals.json and score via hybrid deterministic + LLM checks."""

    def evaluate(
        self,
        agent_def: AgentDefinition | str | Path,
        eval_dataset: EvalDataset | str | Path,
        agent_callable: Callable[[str], AgentResponse] | None = None,
        use_llm: bool | None = None,
        judge_model: str = "gpt-4o-mini",
        output_dir: str | Path | None = None,
    ) -> EvaluationReport:
        agent = self._load_agent(agent_def)
        dataset = self._load_dataset(eval_dataset)

        should_use_llm = has_configured_llm_credentials(judge_model) if use_llm is None else use_llm
        runner = GeneratedAgentRunner(agent) if agent_callable is None else None

        results: list[CaseEvaluationResult] = []
        for case in dataset.cases:
            response = (
                agent_callable(case.question)
                if agent_callable is not None
                else runner.run(case.question, use_llm=should_use_llm)  # type: ignore[union-attr]
            )
            result = self.evaluate_single(
                case=case,
                response=response,
                allowed_sources=agent.allowed_sources,
                use_llm_judge=should_use_llm,
                judge_model=judge_model,
            )
            results.append(result)

        scorecard = self._build_scorecard(agent.agent_name, results)
        mode = "hybrid_deterministic_plus_llm" if should_use_llm else "hybrid_deterministic"
        report = EvaluationReport(
            evaluation_mode=mode,
            scorecard=scorecard,
            results=results,
        )
        report.formatted_scorecard = self.format_scorecard(report)

        if output_dir is not None:
            self.save_outputs(report, output_dir)

        return report

    def evaluate_single(
        self,
        case: EvalCase,
        response: AgentResponse,
        allowed_sources: list[str] | None = None,
        use_llm_judge: bool = False,
        judge_model: str = "gpt-4o-mini",
    ) -> CaseEvaluationResult:
        # 1. Deterministic checks
        valid_json = self._check_valid_contract(response)
        refusal_detected = self._is_refusal_response(response)
        sources_ok = self._check_sources(case, response, allowed_sources or [])
        facts_ok = self._check_expected_facts(case, response)
        grounded_det = self._check_deterministic_groundedness(case, response)

        det_checks: dict[str, bool] = {
            "valid_json_response": valid_json,
            "no_missing_fields": valid_json,
            "correct_refusal_status": (refusal_detected == case.should_refuse),
            "expected_source_exists": sources_ok,
            "required_facts_present": facts_ok,
        }

        failure_reasons: list[str] = []
        llm_notes: str | None = None

        if case.should_refuse:
            grounded: TriState = "N/A"
            correct: TriState = "N/A"
            refusal: TriState = "Yes" if refusal_detected else "No"
            source_quality: TriState = "Yes" if not response.sources else "No"
            if not refusal_detected:
                failure_reasons.append("Agent failed to refuse an unsupported out-of-scope question.")
            if response.sources:
                failure_reasons.append(f"Refused response should not cite sources, got {response.sources}.")
        else:
            if refusal_detected:
                grounded = "No"
                correct = "No"
                refusal = "No"
                source_quality = "No"
                failure_reasons.append("Agent falsely refused an answerable question.")
            else:
                refusal = "N/A"
                grounded = "Yes" if grounded_det else "No"
                correct = "Yes" if facts_ok else "No"
                source_quality = "Yes" if sources_ok else "No"
                if not grounded_det:
                    failure_reasons.append("Answer contains claims/numbers not backed by retrieved evidence.")
                if not facts_ok:
                    failure_reasons.append(f"Answer missed required expected facts: {case.expected_facts}.")
                if not sources_ok:
                    failure_reasons.append(f"Expected sources {case.expected_sources}, got {response.sources}.")

        if not valid_json:
            failure_reasons.append("Response failed output JSON contract validation.")

        # 2. Optional LLM Judge semantic pass
        if use_llm_judge:
            verdict = self._run_llm_judge(case, response, judge_model)
            llm_notes = verdict.reasoning
            if case.should_refuse:
                if not verdict.refusal_appropriate:
                    refusal = "No"
                    failure_reasons.append(f"LLM Judge: {verdict.reasoning}")
            else:
                if not verdict.grounded:
                    grounded = "No"
                    failure_reasons.append(f"LLM Judge ungrounded: {verdict.reasoning}")
                if not verdict.correct:
                    correct = "No"
                    failure_reasons.append(f"LLM Judge incorrect: {verdict.reasoning}")

        passed = self._compute_pass(case.should_refuse, valid_json, grounded, correct, refusal, source_quality)
        overall_score = self._compute_case_score(case.should_refuse, valid_json, grounded, correct, refusal, source_quality)

        return CaseEvaluationResult(
            test_id=case.test_id,
            category=case.category,
            question=case.question,
            should_refuse=case.should_refuse,
            response=response,
            grounded=grounded,
            correct=correct,
            refusal=refusal,
            source_quality=source_quality,
            valid_json_contract=valid_json,
            passed=passed,
            score_label="Pass" if passed else "Fail",
            overall_score=overall_score,
            deterministic_checks=det_checks,
            llm_judge_notes=llm_notes,
            failure_reasons=failure_reasons,
        )

    @staticmethod
    def save_outputs(report: EvaluationReport, output_dir: str | Path) -> dict[str, Path]:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        json_path = out_root / "eval_results.json"
        md_path = out_root / "scorecard.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(report.formatted_scorecard, encoding="utf-8")
        return {"eval_results": json_path, "scorecard": md_path}

    @staticmethod
    def format_scorecard(report: EvaluationReport) -> str:
        sc = report.scorecard
        lines = [
            f"# EVALUATION SCORECARD — {sc.agent_name}",
            "",
            f"- **Total Cases:** {sc.total_cases}",
            f"- **Passed:** {sc.passed_cases} / {sc.total_cases} ({sc.pass_rate:.0%})",
            f"- **Groundedness Rate:** {sc.groundedness_rate:.0%}",
            f"- **Correctness Rate:** {sc.correctness_rate:.0%}",
            f"- **Refusal Accuracy Rate:** {sc.refusal_accuracy_rate:.0%}",
            f"- **Source Quality Rate:** {sc.source_quality_rate:.0%}",
            f"- **JSON Contract Compliance:** {sc.contract_compliance_rate:.0%}",
            "",
            "## Individual Results",
            "| Test | Category | Grounded | Correct | Refusal | Sources | Score |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in report.results:
            lines.append(
                f"| {r.test_id} | {r.category} | {r.grounded} | {r.correct} | "
                f"{r.refusal} | {r.source_quality} | {r.score_label} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_valid_contract(response: AgentResponse) -> bool:
        if not response.answer or not response.answer.strip():
            return False
        if response.confidence not in {"high", "medium", "low", "unsupported"}:
            return False
        if response.raw_json:
            required = {"answer", "sources", "confidence"}
            if not required <= set(response.raw_json.keys()):
                return False
            if not isinstance(response.raw_json["sources"], list):
                return False
        return True

    @staticmethod
    def _is_refusal_response(response: AgentResponse) -> bool:
        if response.refused_unsupported_topic or response.confidence == "unsupported":
            return True
        return any(pat.search(response.answer) for pat in _REFUSAL_PATTERNS)

    @staticmethod
    def _check_sources(case: EvalCase, response: AgentResponse, allowed_sources: list[str]) -> bool:
        if case.should_refuse:
            return len(response.sources) == 0

        if not case.expected_sources:
            return True

        cited = {s.strip().lower() for s in response.sources if s.strip()}
        if not cited:
            return False

        if allowed_sources:
            allowed_lower = {a.strip().lower() for a in allowed_sources}
            if not all(any(c in a or a in c for a in allowed_lower) for c in cited):
                return False

        expected_lower = {e.strip().lower() for e in case.expected_sources}
        matched = sum(1 for exp in expected_lower if any(exp in c or c in exp for c in cited))
        if case.category == "multi_document" and len(expected_lower) >= 2:
            return matched >= 2
        return matched >= 1

    @staticmethod
    def _check_expected_facts(case: EvalCase, response: AgentResponse) -> bool:
        if case.should_refuse:
            return True
        ans_lower = response.answer.lower()
        if not case.expected_facts:
            # Fallback: overlap with expected_answer tokens
            exp_tokens = set(re.findall(r"[a-z0-9]{4,}", case.expected_answer.lower()))
            if not exp_tokens:
                return bool(response.answer.strip())
            ans_tokens = set(re.findall(r"[a-z0-9]{4,}", ans_lower))
            return (len(exp_tokens & ans_tokens) / len(exp_tokens)) >= 0.4

        hits = 0
        for fact in case.expected_facts:
            fact_norm = fact.strip().lower().replace("_", " ")
            has_number = bool(re.search(r"\d", fact_norm))
            matched = fact_norm in ans_lower
            if not matched:
                parts = [p for p in re.findall(r"[a-z0-9]+", fact_norm) if len(p) >= 2 or p.isdigit()]
                matched = bool(parts) and all(p in ans_lower for p in parts)
            if has_number and not matched:
                return False
            if matched:
                hits += 1
        threshold = max(1, (len(case.expected_facts) + 1) // 2)
        return hits >= threshold

    @staticmethod
    def _check_deterministic_groundedness(case: EvalCase, response: AgentResponse) -> bool:
        if case.should_refuse:
            return True
        if not response.retrieved_context:
            return False

        evidence_text = " ".join(response.retrieved_context).lower()
        # Strip doc filename prefixes before extracting answer content tokens
        answer_lower = response.answer.lower()
        ans_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer_lower))
        if ans_numbers and not ans_numbers <= set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text)):
            return False

        ans_words = {
            w for w in re.findall(r"[a-z]{5,}", answer_lower)
            if w not in {"according", "based", "document", "section", "policy", "stated", "available"}
        }
        if not ans_words:
            return True
        ev_words = set(re.findall(r"[a-z]{5,}", evidence_text))
        overlap = len(ans_words & ev_words) / len(ans_words)
        return overlap >= 0.45

    @staticmethod
    def _run_llm_judge(case: EvalCase, response: AgentResponse, model: str) -> LLMJudgeVerdict:
        evidence = "\n".join(response.retrieved_context[:4]) if response.retrieved_context else "None"
        prompt = (
            "Evaluate the AI agent's response strictly against retrieved evidence and ground truth.\n"
            f"Question: {case.question}\n"
            f"Category: {case.category}\n"
            f"Should Refuse: {case.should_refuse}\n"
            f"Expected Answer: {case.expected_answer}\n"
            f"Expected Facts: {case.expected_facts}\n"
            f"Expected Sources: {case.expected_sources}\n\n"
            f"Retrieved Evidence:\n{evidence}\n\n"
            f"Agent Answer: {response.answer}\n"
            f"Agent Sources: {response.sources}\n"
        )
        return complete_structured(prompt=prompt, response_model=LLMJudgeVerdict, model=model, temperature=0.0)

    @staticmethod
    def _compute_pass(
        should_refuse: bool,
        valid_json: bool,
        grounded: TriState,
        correct: TriState,
        refusal: TriState,
        source_quality: TriState,
    ) -> bool:
        if not valid_json:
            return False
        if should_refuse:
            return refusal == "Yes" and source_quality == "Yes"
        return grounded == "Yes" and correct == "Yes" and source_quality == "Yes"

    @staticmethod
    def _compute_case_score(
        should_refuse: bool,
        valid_json: bool,
        grounded: TriState,
        correct: TriState,
        refusal: TriState,
        source_quality: TriState,
    ) -> float:
        if should_refuse:
            checks = [valid_json, refusal == "Yes", source_quality == "Yes"]
        else:
            checks = [valid_json, grounded == "Yes", correct == "Yes", source_quality == "Yes"]
        return round(sum(1 for ok in checks if ok) / len(checks), 2)

    @staticmethod
    def _build_scorecard(agent_name: str, results: list[CaseEvaluationResult]) -> EvaluationScorecard:
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        def rate(values: list[TriState]) -> float:
            applicable = [v for v in values if v != "N/A"]
            if not applicable:
                return 1.0
            return round(sum(1 for v in applicable if v == "Yes") / len(applicable), 2)

        breakdown: dict[str, dict[str, object]] = {}
        for r in results:
            cat_entry = breakdown.setdefault(r.category, {"total": 0, "passed": 0})
            cat_entry["total"] = int(cat_entry["total"]) + 1
            if r.passed:
                cat_entry["passed"] = int(cat_entry["passed"]) + 1
        for cat_entry in breakdown.values():
            t = int(cat_entry["total"])
            p = int(cat_entry["passed"])
            cat_entry["pass_rate"] = round(p / t, 2) if t else 0.0

        return EvaluationScorecard(
            agent_name=agent_name,
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            pass_rate=round(passed / total, 2) if total else 0.0,
            groundedness_rate=rate([r.grounded for r in results]),
            correctness_rate=rate([r.correct for r in results]),
            refusal_accuracy_rate=rate([r.refusal for r in results]),
            source_quality_rate=rate([r.source_quality for r in results]),
            contract_compliance_rate=round(
                sum(1 for r in results if r.valid_json_contract) / total, 2
            ) if total else 0.0,
            category_breakdown=breakdown,
        )

    @staticmethod
    def _load_agent(agent_def: AgentDefinition | str | Path) -> AgentDefinition:
        if isinstance(agent_def, AgentDefinition):
            return agent_def
        return AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))

    @staticmethod
    def _load_dataset(eval_dataset: EvalDataset | str | Path) -> EvalDataset:
        if isinstance(eval_dataset, EvalDataset):
            return eval_dataset
        return EvalDataset.model_validate_json(Path(eval_dataset).read_text(encoding="utf-8"))

