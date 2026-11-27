"""Feature 9 — Honest Final Report generator -> final_report.md."""

from __future__ import annotations

from pathlib import Path

from forgeai.eval.schemas import EvalDataset, EvaluationReport
from forgeai.generator.schemas import AgentDefinition
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.schemas import KnowledgeManifest
from forgeai.optimizer.schemas import OptimizationLog


class FinalReportGenerator:
    """Produce a transparent, evidence-backed quality report summarizing the full ForgeAI run."""

    def generate(
        self,
        brief_text: str,
        requirements: AgentRequirements | str | Path,
        manifest: KnowledgeManifest | str | Path,
        agent_def: AgentDefinition | str | Path,
        eval_dataset: EvalDataset | str | Path,
        initial_report: EvaluationReport | str | Path,
        final_report: EvaluationReport | str | Path,
        optimization_log: OptimizationLog | str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        req = (
            requirements
            if isinstance(requirements, AgentRequirements)
            else AgentRequirements.model_validate_json(Path(requirements).read_text(encoding="utf-8"))
        )
        kb = (
            manifest
            if isinstance(manifest, KnowledgeManifest)
            else KnowledgeManifest.model_validate_json(Path(manifest).read_text(encoding="utf-8"))
        )
        agent = (
            agent_def
            if isinstance(agent_def, AgentDefinition)
            else AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))
        )
        dataset = (
            eval_dataset
            if isinstance(eval_dataset, EvalDataset)
            else EvalDataset.model_validate_json(Path(eval_dataset).read_text(encoding="utf-8"))
        )
        init_rep = (
            initial_report
            if isinstance(initial_report, EvaluationReport)
            else EvaluationReport.model_validate_json(Path(initial_report).read_text(encoding="utf-8"))
        )
        fin_rep = (
            final_report
            if isinstance(final_report, EvaluationReport)
            else EvaluationReport.model_validate_json(Path(final_report).read_text(encoding="utf-8"))
        )
        opt_log = (
            optimization_log
            if isinstance(optimization_log, OptimizationLog)
            else OptimizationLog.model_validate_json(Path(optimization_log).read_text(encoding="utf-8"))
        )

        failed_cases = [r for r in fin_rep.results if not r.passed]
        weaknesses = self._identify_weaknesses(agent, fin_rep, failed_cases)
        recommendations = self._generate_recommendations(agent, fin_rep, failed_cases, kb)

        lines: list[str] = [
            "# FORGEAI QUALITY REPORT",
            "",
            f"**Agent:** {agent.agent_name}",
            f"**Domain:** {agent.domain}",
            f"**Target Audience:** {agent.target_audience}",
            f"**Purpose:** {agent.purpose}",
            f"**Documents Analyzed:** {kb.total_documents} ({kb.total_chunks} chunks)",
            f"**Generated Test Cases:** {dataset.total_cases}",
            "",
            "## 1. Input Brief",
            f"> {brief_text.strip()}",
            "",
            "## 2. Documents Analyzed",
            "| Document | Type | Chunks | Sections |",
            "| --- | --- | ---: | --- |",
        ]
        for doc in kb.documents:
            secs = ", ".join(doc.sections_detected[:4]) or "-"
            lines.append(f"| {doc.document_name} | {doc.document_type} | {doc.chunk_count} | {secs} |")

        lines.extend(
            [
                "",
                "## 3. Topic Scope & Groundability",
                f"- **Supported Topics:** {', '.join(agent.supported_topics) or 'None'}",
                f"- **Partially Supported Topics:** {', '.join(agent.partially_supported_topics) or 'None'}",
                f"- **Unsupported Topics (Refused):** {', '.join(agent.unsupported_topics) or 'None'}",
                "",
                "## 4. Selected Tools",
            ]
        )
        for tool in agent.tools:
            lines.append(f"- **`{tool.name}`** (`{tool.type}`): {tool.selection_reason}")
        if agent.rejected_tools:
            lines.append("- **Excluded Tools:** " + ", ".join(f"`{t['name']}` ({t['reason']})" for t in agent.rejected_tools))

        rc = agent.retrieval_config
        lines.extend(
            [
                "",
                "## 5. Retrieval Configuration",
                f"- **Strategy:** `{rc.strategy}`",
                f"- **Chunk Size / Overlap:** `{rc.chunk_size}` / `{rc.chunk_overlap}`",
                f"- **Top-K:** `{rc.top_k}`",
                f"- **Hybrid Keyword Weight:** `{rc.hybrid_keyword_weight}`",
                "",
                "## 6. Generated System Prompt",
                "```text",
                agent.system_prompt,
                "```",
                "",
                "## 7. Evaluation & Optimization Trajectory",
                f"- **Evaluation Dataset Size:** {dataset.total_cases} cases ({dataset.category_counts})",
                "- **Benchmark Scope Note:** These metrics reflect the internal evaluation pass rate on "
                "document-derived test cases (`evals.json`) and should not be interpreted as generalized "
                "real-world statistical accuracy.",
                f"- **INITIAL EVALUATION PASS RATE:** {init_rep.scorecard.passed_cases}/{init_rep.scorecard.total_cases} "
                f"({init_rep.scorecard.pass_rate:.0%}) | "
                f"Groundedness: {init_rep.scorecard.groundedness_rate:.0%}, "
                f"Correctness: {init_rep.scorecard.correctness_rate:.0%}, "
                f"Refusal: {init_rep.scorecard.refusal_accuracy_rate:.0%}, "
                f"Sources: {init_rep.scorecard.source_quality_rate:.0%}",
            ]
        )

        total_regressions = sum(len(rec.regressed_tests) for rec in opt_log.iterations)
        if opt_log.iterations:
            for rec in opt_log.iterations:
                lines.append(
                    f"- **ITERATION {rec.iteration} ({rec.failure_type}):** "
                    f"Pass Rate: {rec.passed_after}/{opt_log.total_cases} "
                    f"({rec.score_before:.0%} -> {rec.score_after:.0%}) | "
                    f"Change: {rec.change_applied}"
                    + (" [Rolled back]" if rec.rolled_back else "")
                )
        else:
            lines.append("- **Optimization Iterations:** 0 iterations required.")

        lines.extend(
            [
                f"- **FINAL EVALUATION PASS RATE:** {fin_rep.scorecard.passed_cases}/{fin_rep.scorecard.total_cases} "
                f"({fin_rep.scorecard.pass_rate:.0%}) | "
                f"Regressions reported: {total_regressions} | Stop reason: {opt_log.stop_reason}",
                "",
                "## 8. Failed Test Cases (Honest Disclosure)",
            ]
        )
        if failed_cases:
            lines.append("| Test ID | Category | Question | Failure Reasons |")
            lines.append("| --- | --- | --- | --- |")
            for fc in failed_cases:
                reasons = "; ".join(fc.failure_reasons) or "Did not meet evaluation threshold."
                lines.append(f"| {fc.test_id} | {fc.category} | {fc.question} | {reasons} |")
        else:
            lines.append("- None. All evaluation test cases passed.")

        lines.extend(["", "## 9. Remaining Weaknesses"])
        lines.extend(f"- {w}" for w in weaknesses)

        lines.extend(["", "## 10. Recommendations"])
        lines.extend(f"- {rec}" for rec in recommendations)

        markdown = "\n".join(lines) + "\n"
        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(markdown, encoding="utf-8")
        return markdown

    @staticmethod
    def _identify_weaknesses(
        agent: AgentDefinition,
        fin_rep: EvaluationReport,
        failed_cases: list,
    ) -> list[str]:
        weaknesses: list[str] = []
        if failed_cases:
            cats = sorted({f.category for f in failed_cases})
            weaknesses.append(
                f"{len(failed_cases)} evaluation test case(s) still fail across categories: {', '.join(cats)}."
            )
        if fin_rep.scorecard.correctness_rate < 1.0:
            weaknesses.append(f"Correctness rate is {fin_rep.scorecard.correctness_rate:.0%} (< 100%).")
        if fin_rep.scorecard.source_quality_rate < 1.0:
            weaknesses.append(f"Source citation quality is {fin_rep.scorecard.source_quality_rate:.0%} (< 100%).")
        if agent.partially_supported_topics:
            weaknesses.append(
                f"Partially supported topics ({', '.join(agent.partially_supported_topics)}) lack full policy depth in uploaded docs."
            )
        if agent.unsupported_topics:
            weaknesses.append(
                f"Unsupported topics ({', '.join(agent.unsupported_topics)}) cannot be answered without additional data sources."
            )
        if not weaknesses:
            weaknesses.append("No major weaknesses detected within the current document scope.")
        return weaknesses

    @staticmethod
    def _generate_recommendations(
        agent: AgentDefinition,
        fin_rep: EvaluationReport,
        failed_cases: list,
        kb: KnowledgeManifest,
    ) -> list[str]:
        recs: list[str] = []
        failed_cats = {f.category for f in failed_cases}
        if "multi_document" in failed_cats:
            recs.append("Improve cross-document retrieval and enable LLM-based multi-document synthesis.")
        if "ambiguous" in failed_cats:
            recs.append("Add proactive clarification prompts for underspecified user questions.")
        if agent.partially_supported_topics:
            recs.append(
                f"Upload complete policy documents for partially supported topics: {', '.join(agent.partially_supported_topics)}."
            )
        if agent.unsupported_topics:
            recs.append(
                f"Connect authorized data sources or keep strict refusals enabled for: {', '.join(agent.unsupported_topics)}."
            )
        if kb.embedding_provider == "local":
            recs.append("Configure an API or neural embedding model (e.g., LiteLLM text-embedding-3-small) for production semantic recall.")
        if not recs:
            recs.append("Agent is ready for deployment; monitor production queries and expand evals.json regularly.")
        return recs

