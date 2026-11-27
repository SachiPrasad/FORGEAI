"""Critic & Optimizer Loop (maximum 3 iterations, regression tracking, targeted fixes)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

from forgeai.eval.dataset_generator import EvalDatasetGenerator
from forgeai.eval.judge import EvaluationJudge
from forgeai.eval.schemas import AgentResponse, EvalDataset, EvaluationReport
from forgeai.generator.schemas import AgentDefinition, ToolSpec
from forgeai.generator.validator import AgentDefinitionValidator
from forgeai.optimizer.critic import FailureAnalyzer
from forgeai.optimizer.schemas import (
    MAX_OPTIMIZATION_ITERATIONS,
    FailureDiagnosis,
    FailureType,
    IterationRecord,
    OptimizationLog,
)
from forgeai.tools.registry import build_default_registry


class CriticOptimizerLoop:
    """Iteratively evaluate, diagnose failures, apply targeted fixes, and re-verify (max 3 iterations)."""

    def __init__(
        self,
        analyzer: FailureAnalyzer | None = None,
        judge: EvaluationJudge | None = None,
        validator: AgentDefinitionValidator | None = None,
    ) -> None:
        self.analyzer = analyzer or FailureAnalyzer()
        self.judge = judge or EvaluationJudge()
        self.validator = validator or AgentDefinitionValidator()

    def optimize(
        self,
        agent_def: AgentDefinition | str | Path,
        eval_dataset: EvalDataset | str | Path,
        max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
        use_llm: bool | None = False,
        agent_callable_factory: Callable[[AgentDefinition], Callable[[str], AgentResponse]] | None = None,
        output_dir: str | Path | None = None,
        save_optimized_agent_path: str | Path | None = None,
    ) -> tuple[AgentDefinition, OptimizationLog, EvaluationReport, EvaluationReport]:
        """Run up to 3 optimization iterations and return (best_agent, log, initial_report, final_report)."""
        capped_iters = min(max(1, max_iterations), MAX_OPTIMIZATION_ITERATIONS)
        current_agent = self._load_agent(agent_def)
        dataset = self._load_dataset(eval_dataset)

        initial_report = self._run_eval(current_agent, dataset, use_llm, agent_callable_factory)
        best_agent = current_agent
        best_report = initial_report
        iteration_scorecards: list[dict[str, object]] = [
            {
                "iteration": 0,
                "label": "Initial Evaluation",
                "scorecard": initial_report.scorecard.model_dump(),
            }
        ]

        iterations: list[IterationRecord] = []
        attempted_fixes: set[str] = set()
        stop_reason = f"Reached maximum iteration limit ({capped_iters})."

        for step in range(1, capped_iters + 1):
            if best_report.scorecard.pass_rate >= 1.0:
                stop_reason = "All evaluation test cases passed (100% pass rate)."
                break

            diagnoses = self.analyzer.analyze(best_report, dataset, best_agent)
            if not diagnoses:
                stop_reason = "No actionable failure diagnoses remained."
                break

            target_type, target_diags = self._select_next_target(diagnoses, attempted_fixes)
            if target_type is None:
                stop_reason = "All applicable targeted fix strategies have already been attempted."
                break

            candidate_agent, fix_key, change_desc, reason_desc = self._apply_targeted_fix(
                best_agent,
                failure_type=target_type,
                diagnoses=target_diags,
                attempted_fixes=attempted_fixes,
            )
            if candidate_agent is None:
                attempted_fixes.add(f"{target_type}:exhausted")
                stop_reason = "No further configuration improvements available."
                continue

            attempted_fixes.add(fix_key)
            candidate_agent.validation = self.validator.validate(
                candidate_agent,
                has_knowledge_base=bool(candidate_agent.retrieval_config.knowledge_manifest_path),
            )

            new_report = self._run_eval(candidate_agent, dataset, use_llm, agent_callable_factory)
            prev_passed_ids = {r.test_id for r in best_report.results if r.passed}
            new_passed_ids = {r.test_id for r in new_report.results if r.passed}

            newly_passed = sorted(new_passed_ids - prev_passed_ids)
            regressed = sorted(prev_passed_ids - new_passed_ids)

            score_before = best_report.scorecard.pass_rate
            score_after = new_report.scorecard.pass_rate
            rolled_back = score_after < score_before

            iterations.append(
                IterationRecord(
                    iteration=step,
                    failure_type=target_type,
                    reason=reason_desc,
                    change_applied=change_desc,
                    tests_addressed=[d.test_id for d in target_diags],
                    score_before=score_before,
                    score_after=score_after,
                    passed_before=best_report.scorecard.passed_cases,
                    passed_after=new_report.scorecard.passed_cases,
                    newly_passed_tests=newly_passed,
                    regressed_tests=regressed,
                    rolled_back=rolled_back,
                )
            )

            if not rolled_back:
                best_agent = candidate_agent
                best_report = new_report

            iteration_scorecards.append(
                {
                    "iteration": step,
                    "label": f"Iteration {step} ({target_type})",
                    "rolled_back": rolled_back,
                    "scorecard": new_report.scorecard.model_dump(),
                }
            )

            if best_report.scorecard.pass_rate >= 1.0:
                stop_reason = "All evaluation test cases passed (100% pass rate)."
                break

        log = OptimizationLog(
            agent_name=best_agent.agent_name,
            max_iterations=capped_iters,
            total_iterations_run=len(iterations),
            initial_score=initial_report.scorecard.pass_rate,
            final_score=best_report.scorecard.pass_rate,
            initial_passed=initial_report.scorecard.passed_cases,
            final_passed=best_report.scorecard.passed_cases,
            total_cases=best_report.scorecard.total_cases,
            stop_reason=stop_reason,
            iterations=iterations,
            metadata={
                "initial_groundedness": initial_report.scorecard.groundedness_rate,
                "final_groundedness": best_report.scorecard.groundedness_rate,
                "initial_correctness": initial_report.scorecard.correctness_rate,
                "final_correctness": best_report.scorecard.correctness_rate,
            },
        )

        if output_dir is not None:
            import json

            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "iteration_log.json").write_text(log.model_dump_json(indent=2), encoding="utf-8")
            (out_root / "scorecards.json").write_text(
                json.dumps(
                    {
                        "agent_name": best_agent.agent_name,
                        "initial_score": log.initial_score,
                        "final_score": log.final_score,
                        "scorecards": iteration_scorecards,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.judge.save_outputs(best_report, out_root)
            self._save_version_history(
                initial_agent=current_agent,
                best_agent=best_agent,
                iterations=iterations,
                versions_dir=out_root / "versions",
            )

        if save_optimized_agent_path is not None:
            from forgeai.generator.generator import AgentGenerator

            AgentGenerator.save_agent_definition(best_agent, save_optimized_agent_path)

        return best_agent, log, initial_report, best_report

    def _run_eval(
        self,
        agent: AgentDefinition,
        dataset: EvalDataset,
        use_llm: bool | None,
        callable_factory: Callable[[AgentDefinition], Callable[[str], AgentResponse]] | None,
    ) -> EvaluationReport:
        custom_callable = callable_factory(agent) if callable_factory is not None else None
        return self.judge.evaluate(
            agent_def=agent,
            eval_dataset=dataset,
            agent_callable=custom_callable,
            use_llm=use_llm,
        )

    @staticmethod
    def _select_next_target(
        diagnoses: list[FailureDiagnosis],
        attempted_fixes: set[str],
    ) -> tuple[FailureType | None, list[FailureDiagnosis]]:
        counts = Counter(d.failure_type for d in diagnoses)
        # Prioritize systemic fixes first: tool_failure -> retrieval_miss -> wrong_refusal -> hallucination -> prompt_defect -> output_failure
        priority_order: list[FailureType] = [
            "tool_failure",
            "retrieval_miss",
            "wrong_refusal",
            "hallucination",
            "prompt_defect",
            "output_failure",
        ]
        # Sort by (count desc, priority asc)
        ranked = sorted(
            counts.keys(),
            key=lambda ft: (-counts[ft], priority_order.index(ft)),
        )
        for ft in ranked:
            if f"{ft}:exhausted" not in attempted_fixes:
                return ft, [d for d in diagnoses if d.failure_type == ft]
        return None, []

    def _apply_targeted_fix(
        self,
        agent: AgentDefinition,
        failure_type: FailureType,
        diagnoses: list[FailureDiagnosis],
        attempted_fixes: set[str],
    ) -> tuple[AgentDefinition | None, str, str, str]:
        updated = agent.model_copy(deep=True)
        reason = diagnoses[0].root_cause

        if failure_type == "tool_failure":
            fix_key = "tool_failure:attach_retriever"
            if fix_key not in attempted_fixes:
                registry = build_default_registry()
                entry = registry.get_entry("document_retriever")
                tools = [t for t in updated.tools if t.name != "document_retriever"]
                tools.insert(
                    0,
                    ToolSpec(
                        name="document_retriever",
                        type=entry.category,
                        description=entry.description,
                        enabled=True,
                        selection_reason="Restored by Critic & Optimizer Loop after tool_failure diagnosis.",
                        parameters={
                            "manifest_path": updated.retrieval_config.knowledge_manifest_path,
                            "top_k": updated.retrieval_config.top_k,
                        },
                    ),
                )
                updated.tools = tools
                return (
                    updated,
                    fix_key,
                    "Attached and enabled 'document_retriever' tool from ToolRegistry.",
                    reason,
                )

        if failure_type == "retrieval_miss":
            fix_key_1 = "retrieval_miss:expand_topk_hybrid"
            if fix_key_1 not in attempted_fixes:
                old_k = updated.retrieval_config.top_k
                old_w = updated.retrieval_config.hybrid_keyword_weight
                new_k = min(12, max(old_k + 4, 8))
                new_w = max(old_w, 0.45)
                updated.retrieval_config = updated.retrieval_config.model_copy(
                    update={"top_k": new_k, "hybrid_keyword_weight": new_w, "min_score": -1.0}
                )
                for t in updated.tools:
                    if t.name == "document_retriever":
                        t.parameters["top_k"] = new_k
                return (
                    updated,
                    fix_key_1,
                    f"Increased retrieval_config.top_k from {old_k} to {new_k} and hybrid_keyword_weight from {old_w} to {new_w}.",
                    reason,
                )

        if failure_type == "wrong_refusal":
            fix_key = "wrong_refusal:clarify_policy_vs_personal"
            if fix_key not in attempted_fixes:
                old_k = updated.retrieval_config.top_k
                new_k = min(12, max(old_k + 2, 8))
                updated.retrieval_config = updated.retrieval_config.model_copy(update={"top_k": new_k})
                updated.system_prompt += (
                    "\n\nClarification on escalation & policy rules:\n"
                    "- Answer documented policy escalation procedures (such as where to escalate payroll or contract disputes) "
                    "using the approved handbook, while refusing requests for personal salary or balance data."
                )
                return (
                    updated,
                    fix_key,
                    f"Clarified documented escalation vs. personal-data refusal boundary and set top_k={new_k}.",
                    reason,
                )

        if failure_type == "prompt_defect":
            fix_key_1 = "prompt_defect:multi_doc_detail_mode"
            if fix_key_1 not in attempted_fixes:
                updated.system_prompt += (
                    "\n\nMulti-document synthesis & completeness directive (detail_mode: full):\n"
                    "- Include all relevant facts, numbers, and conditions from every matching retrieved document.\n"
                    "- Always cite every contributing document in `sources`."
                )
                return (
                    updated,
                    fix_key_1,
                    "Updated system_prompt with full multi-document synthesis and complete fact-citation instructions (detail_mode: full).",
                    reason,
                )

        if failure_type == "hallucination":
            fix_key = "hallucination:strict_grounding_temp0"
            if fix_key not in attempted_fixes:
                updated.model_config_details = updated.model_config_details.model_copy(update={"temperature": 0.0})
                guardrail = "Never alter or extrapolate numbers/dates; repeat only exact figures from retrieved context."
                if guardrail not in updated.restrictions:
                    updated.restrictions.append(guardrail)
                updated.system_prompt += f"\n\nStrict Grounding Addendum:\n- {guardrail}"
                return (
                    updated,
                    fix_key,
                    "Set model temperature=0.0 and added strict verbatim-number & refusal grounding rules to system_prompt.",
                    reason,
                )

        if failure_type == "output_failure":
            fix_key = "output_failure:enforce_contract"
            if fix_key not in attempted_fixes:
                updated.system_prompt += (
                    "\n\nStrict JSON Contract Enforcement:\n"
                    "- Always return valid JSON containing keys: answer (string), sources (array of strings), confidence (string)."
                )
                return (
                    updated,
                    fix_key,
                    "Added strict JSON output_contract enforcement instructions to system_prompt.",
                    reason,
                )

        return None, f"{failure_type}:exhausted", "", reason

    @staticmethod
    def _save_version_history(
        initial_agent: AgentDefinition,
        best_agent: AgentDefinition,
        iterations: list[IterationRecord],
        versions_dir: Path,
    ) -> None:
        import difflib
        import json

        from forgeai.generator.generator import AgentGenerator

        versions_dir.mkdir(parents=True, exist_ok=True)
        AgentGenerator.save_agent_definition(initial_agent, versions_dir / "v0_initial.json")
        AgentGenerator.save_agent_definition(best_agent, versions_dir / f"v{len(iterations)}_final.json")

        init_cfg_lines = json.dumps(
            {
                "retrieval_config": initial_agent.retrieval_config.model_dump(),
                "model_config": initial_agent.model_config_details.model_dump(),
            },
            indent=2,
        ).splitlines()
        final_cfg_lines = json.dumps(
            {
                "retrieval_config": best_agent.retrieval_config.model_dump(),
                "model_config": best_agent.model_config_details.model_dump(),
            },
            indent=2,
        ).splitlines()
        config_diff_lines = list(
            difflib.unified_diff(
                init_cfg_lines,
                final_cfg_lines,
                fromfile="v0_initial/config",
                tofile=f"v{len(iterations)}_final/config",
                lineterm="",
            )
        )

        prompt_diff_lines = list(
            difflib.unified_diff(
                initial_agent.system_prompt.splitlines(),
                best_agent.system_prompt.splitlines(),
                fromfile="v0_initial/system_prompt",
                tofile=f"v{len(iterations)}_final/system_prompt",
                lineterm="",
            )
        )

        total_regressions = sum(len(rec.regressed_tests) for rec in iterations)
        diff_summary = {
            "initial_version": "v0",
            "final_version": f"v{len(iterations)}",
            "regression_count": total_regressions,
            "changes": [
                {
                    "iteration": rec.iteration,
                    "failure_type": rec.failure_type,
                    "change_applied": rec.change_applied,
                    "score_before": rec.score_before,
                    "score_after": rec.score_after,
                    "newly_passed_tests": rec.newly_passed_tests,
                    "regressed_tests": rec.regressed_tests,
                    "rolled_back": rec.rolled_back,
                }
                for rec in iterations
            ],
            "retrieval_top_k_changed": (
                f"{initial_agent.retrieval_config.top_k} -> {best_agent.retrieval_config.top_k}"
            ),
            "hybrid_keyword_weight_changed": (
                f"{initial_agent.retrieval_config.hybrid_keyword_weight} -> "
                f"{best_agent.retrieval_config.hybrid_keyword_weight}"
            ),
            "temperature_changed": (
                f"{initial_agent.model_config_details.temperature} -> {best_agent.model_config_details.temperature}"
            ),
            "system_prompt_length_changed": f"{len(initial_agent.system_prompt)} -> {len(best_agent.system_prompt)}",
            "config_diff": "\n".join(config_diff_lines) if config_diff_lines else "No config parameter diff.",
            "prompt_diff": "\n".join(prompt_diff_lines) if prompt_diff_lines else "No system prompt diff.",
        }
        (versions_dir / "version_diff.json").write_text(json.dumps(diff_summary, indent=2), encoding="utf-8")

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

