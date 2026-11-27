"""Unified end-to-end FORGEAI pipeline connecting Features 1–10 + observability, versioning, export, & dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from forgeai.eval import EvalDataset, EvalDatasetGenerator, EvaluationReport
from forgeai.exporter import AgentExporter
from forgeai.generator import AgentDefinition, AgentGenerator
from forgeai.groundability import GroundabilityGuard, GroundabilityReport, SupportedTopicConfig
from forgeai.intake import AgentRequirements, IntakeInterviewer, UserConfig
from forgeai.knowledge import ChunkingConfig, KnowledgeManifest, KnowledgeProcessor
from forgeai.observability import TraceCollector, TraceSummary
from forgeai.observability.logger import get_forgeai_logger
from forgeai.optimizer import CriticOptimizerLoop, MAX_OPTIMIZATION_ITERATIONS, OptimizationLog
from forgeai.reporting import FinalReportGenerator, VisualDashboardGenerator
from forgeai.runtime import AgentRuntime
from forgeai.tools import DynamicToolSelector, ToolSelectionRecord


class ForgeAIPipelineResult:
    """Container holding all outputs produced across Features 1–10 plus observability and export."""

    def __init__(
        self,
        requirements: AgentRequirements,
        manifest: KnowledgeManifest,
        groundability_report: GroundabilityReport,
        supported_topic_config: SupportedTopicConfig,
        tool_selection: ToolSelectionRecord,
        agent: AgentDefinition,
        eval_dataset: EvalDataset,
        initial_eval_report: EvaluationReport,
        final_eval_report: EvaluationReport,
        optimization_log: OptimizationLog,
        final_report_markdown: str,
        runtime: AgentRuntime,
        trace_summary: TraceSummary,
        dashboard_html: str,
        export_paths: dict[str, Path],
    ) -> None:
        self.requirements = requirements
        self.manifest = manifest
        self.groundability_report = groundability_report
        self.supported_topic_config = supported_topic_config
        self.tool_selection = tool_selection
        self.agent = agent
        self.eval_dataset = eval_dataset
        self.initial_eval_report = initial_eval_report
        self.final_eval_report = final_eval_report
        self.optimization_log = optimization_log
        self.final_report_markdown = final_report_markdown
        self.runtime = runtime
        self.trace_summary = trace_summary
        self.dashboard_html = dashboard_html
        self.export_paths = export_paths


class ForgeAIPipeline:
    """Run all 10 FORGEAI features plus tracing, versioning, export bundle, and visual dashboard."""

    def run(
        self,
        brief: str | Path,
        docs_dir: str | Path,
        output_dir: str | Path,
        config: UserConfig | dict[str, Any] | str | Path | None = None,
        extra_topics: list[str] | None = None,
        model: str = "gpt-4o-mini",
        use_llm: bool | None = None,
        max_opt_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
    ) -> ForgeAIPipelineResult:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        tracer = TraceCollector(model=model)
        logger = get_forgeai_logger(log_dir=out_root / "logs")
        logger.info(f"Starting FORGEAI pipeline run -> {out_root}")

        brief_path = Path(brief) if isinstance(brief, (str, Path)) and Path(str(brief)).is_file() else None
        brief_text = brief_path.read_text(encoding="utf-8") if brief_path else str(brief)

        # Feature 1 — Intelligent Intake Interviewer
        req_path = out_root / "requirements.json"
        with tracer.span("feature_1_intake", "extract_requirements", is_llm_call=bool(use_llm)) as span_ctx:
            requirements = IntakeInterviewer().run(
                brief=brief,
                docs_dir=docs_dir,
                config=config,
                output_path=req_path,
            )
            span_ctx["input_tokens"] = tracer.estimate_tokens(brief_text)
            span_ctx["output_tokens"] = tracer.estimate_tokens(requirements.model_dump_json())

        # Feature 2 — Document Ingestion & Knowledge Processing
        kb_dir = out_root / "kb"
        with tracer.span("feature_2_knowledge", "ingest_documents") as span_ctx:
            manifest = KnowledgeProcessor().process(
                docs_dir=docs_dir,
                output_dir=kb_dir,
                chunking=ChunkingConfig(strategy="recursive", chunk_size=600, chunk_overlap=100),
                embedding_provider="local" if use_llm is False else "auto",
            )
            span_ctx["attributes"] = {
                "total_documents": manifest.total_documents,
                "total_chunks": manifest.total_chunks,
            }
        manifest_path = kb_dir / "knowledge_manifest.json"

        # Feature 3 — Groundability Guard
        ground_dir = out_root / "groundability"
        with tracer.span("feature_3_groundability", "evaluate_groundability", is_llm_call=bool(use_llm)) as span_ctx:
            ground_report, topic_cfg = GroundabilityGuard().evaluate(
                requirements=requirements,
                manifest=manifest_path,
                extra_topics=extra_topics,
                use_llm=use_llm,
                model=model,
                output_dir=ground_dir,
            )
            span_ctx["attributes"] = {"groundability_score": ground_report.groundability_score}

        # Feature 5 — Tool Registry & Dynamic Tool Selection
        tools_dir = out_root / "tools"
        with tracer.span("feature_5_tools", "select_tools") as span_ctx:
            tool_selection = DynamicToolSelector().select(
                requirements=requirements,
                topic_config=topic_cfg,
                manifest=manifest,
                manifest_path=manifest_path,
                use_llm=use_llm,
                model=model,
                output_dir=tools_dir,
            )
            span_ctx["attributes"] = {"selected_tools": tool_selection.selected_tools}

        # Feature 4 — Agent Generator
        agent_path = out_root / "agent.json"
        with tracer.span("feature_4_generator", "generate_agent_json", is_llm_call=bool(use_llm)) as span_ctx:
            agent_def = AgentGenerator().generate(
                requirements=requirements,
                supported_topic_config=topic_cfg,
                knowledge_manifest=manifest_path,
                model=model,
                use_llm=use_llm,
                output_path=agent_path,
            )
            span_ctx["output_tokens"] = tracer.estimate_tokens(agent_def.system_prompt)

        # Feature 6 — Automatic Evaluation Dataset Generator (>= 20 cases)
        eval_dir = out_root / "eval"
        evals_path = eval_dir / "evals.json"
        with tracer.span("feature_6_eval_gen", "generate_evals", is_llm_call=bool(use_llm)) as span_ctx:
            eval_dataset = EvalDatasetGenerator().generate(
                agent_def=agent_def,
                manifest=manifest,
                use_llm=use_llm,
                model=model,
                output_path=evals_path,
            )
            span_ctx["attributes"] = {"total_cases": eval_dataset.total_cases}

        # Features 7 & 8 — Evaluation Judge + Critic & Optimizer Loop (max 3 iterations + versioning)
        with tracer.span("feature_7_8_eval_opt", "critic_optimizer_loop", is_llm_call=bool(use_llm)) as span_ctx:
            best_agent, opt_log, init_report, final_report = CriticOptimizerLoop().optimize(
                agent_def=agent_def,
                eval_dataset=eval_dataset,
                max_iterations=max_opt_iterations,
                use_llm=use_llm,
                output_dir=eval_dir,
                save_optimized_agent_path=agent_path,
            )
            span_ctx["attributes"] = {
                "initial_score": opt_log.initial_score,
                "final_score": opt_log.final_score,
                "iterations": opt_log.total_iterations_run,
            }

        # Feature 9 — Honest Final Report
        with tracer.span("feature_9_reporting", "generate_final_report"):
            final_report_md = FinalReportGenerator().generate(
                brief_text=brief_text,
                requirements=requirements,
                manifest=manifest,
                agent_def=best_agent,
                eval_dataset=eval_dataset,
                initial_report=init_report,
                final_report=final_report,
                optimization_log=opt_log,
                output_path=out_root / "final_report.md",
            )

        # Export Bundle + Observability Trace + Visual Dashboard + Canonical Root Artifacts (Section 8)
        with tracer.span("export_bundle", "export_standalone_agent"):
            export_paths = AgentExporter().export(best_agent, output_dir=out_root / "export_bundle")

        self._sync_canonical_root_artifacts(out_root, manifest)

        trace_summary = tracer.summary()
        tracer.save_json(out_root / "trace_log.json")

        version_diff_data: dict[str, Any] | None = None
        diff_file = eval_dir / "versions" / "version_diff.json"
        if diff_file.is_file():
            import json

            version_diff_data = json.loads(diff_file.read_text(encoding="utf-8"))

        dashboard_html = VisualDashboardGenerator().generate(
            agent_def=best_agent,
            groundability_report=ground_report,
            initial_report=init_report,
            final_report=final_report,
            optimization_log=opt_log,
            requirements=requirements,
            manifest=manifest,
            eval_dataset=eval_dataset,
            trace_summary=trace_summary,
            version_diff=version_diff_data,
            output_path=out_root / "dashboard.html",
        )
        logger.info(
            f"Completed pipeline for {best_agent.agent_name}: "
            f"{init_report.scorecard.pass_rate:.0%} -> {final_report.scorecard.pass_rate:.0%}"
        )

        # Feature 10 — Working Generated Agent Runtime
        runtime = AgentRuntime(best_agent)

        return ForgeAIPipelineResult(
            requirements=requirements,
            manifest=manifest,
            groundability_report=ground_report,
            supported_topic_config=topic_cfg,
            tool_selection=tool_selection,
            agent=best_agent,
            eval_dataset=eval_dataset,
            initial_eval_report=init_report,
            final_eval_report=final_report,
            optimization_log=opt_log,
            final_report_markdown=final_report_md,
            runtime=runtime,
            trace_summary=trace_summary,
            dashboard_html=dashboard_html,
            export_paths=export_paths,
        )

    @staticmethod
    def _sync_canonical_root_artifacts(out_root: Path, manifest: KnowledgeManifest) -> None:
        """Ensure all 8 canonical Section-8 artifact files exist cleanly at out_root."""
        import shutil

        copies = [
            (out_root / "groundability" / "groundability_report.json", out_root / "groundability_report.json"),
            (out_root / "eval" / "evals.json", out_root / "evals.json"),
            (out_root / "eval" / "scorecards.json", out_root / "scorecards.json"),
            (out_root / "eval" / "iteration_log.json", out_root / "iteration_log.json"),
        ]
        for src, dst in copies:
            if src.is_file():
                shutil.copy2(src, dst)

        root_manifest = manifest.model_copy(update={"vector_store_path": "kb/knowledge_base.sqlite"})
        (out_root / "knowledge_manifest.json").write_text(root_manifest.model_dump_json(indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="forgeai-run", description="Run the full 10-feature FORGEAI pipeline.")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)

    result = ForgeAIPipeline().run(
        brief=args.brief,
        docs_dir=args.docs,
        output_dir=args.out,
        config=args.config,
        extra_topics=args.topics,
        model=args.model,
        use_llm=False if args.no_llm else None,
    )
    console = Console()
    console.print(
        f"[bold green]FORGEAI Pipeline Complete[/bold green]\n"
        f"- Agent: {result.agent.agent_name}\n"
        f"- Initial Score: {result.initial_eval_report.scorecard.passed_cases}/{result.eval_dataset.total_cases} "
        f"({result.initial_eval_report.scorecard.pass_rate:.0%})\n"
        f"- Final Score: {result.final_eval_report.scorecard.passed_cases}/{result.eval_dataset.total_cases} "
        f"({result.final_eval_report.scorecard.pass_rate:.0%})\n"
        f"- Artifacts saved in: {Path(args.out).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
