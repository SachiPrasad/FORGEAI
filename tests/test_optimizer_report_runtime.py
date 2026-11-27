"""Comprehensive tests for Features 8, 9, 10, and the unified 10-feature ForgeAIPipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.eval import AgentResponse, EvalCase, EvalDatasetGenerator, EvaluationJudge
from forgeai.generator import AgentGenerator
from forgeai.groundability import GroundabilityGuard
from forgeai.intake import IntakeInterviewer
from forgeai.knowledge import KnowledgeProcessor
from forgeai.optimizer import MAX_OPTIMIZATION_ITERATIONS, CriticOptimizerLoop, FailureAnalyzer
from forgeai.pipeline import ForgeAIPipeline
from forgeai.reporting import FinalReportGenerator
from forgeai.runtime import AgentRuntime, ConversationMemory
from forgeai.runtime.cli import main as runtime_cli_main


@pytest.fixture()
def full_hr_docs(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.txt").write_text(
        "Annual Leave:\nFull-time employees receive 20 days of annual paid leave and 10 paid sick days per year.\n"
        "Parental leave provides up to 12 weeks of paid time off after 6 months of tenure.\n"
        "Unused annual leave up to 5 days may carry over into Q1.",
        encoding="utf-8",
    )
    (docs / "attendance_rules.csv").write_text(
        "rule_id,category,rule_summary,notice_required\n"
        "ATT-01,Core Hours,Core collaboration window is 10:00 AM to 4:00 PM,N/A\n"
        "ATT-02,Remote Work,Up to 2 remote work days per week,24 hours\n",
        encoding="utf-8",
    )
    (docs / "hr_handbook.txt").write_text(
        "Escalation & Exceptions:\n"
        "Policy exceptions, payroll disputes, and individual contract questions must be escalated to HR Operations.\n",
        encoding="utf-8",
    )
    brief = tmp_path / "brief.txt"
    brief.write_text(
        "Build an HR assistant for employees that answers questions about leave policies, attendance, and company rules.",
        encoding="utf-8",
    )
    return brief, docs


def test_critic_optimizer_enforces_max_3_iterations_and_improves_score(
    full_hr_docs: tuple[Path, Path], tmp_path: Path
):
    brief, docs = full_hr_docs
    req = IntakeInterviewer().run(brief=brief, docs_dir=docs, config={"use_llm": False})
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    _, topic_cfg = GroundabilityGuard().evaluate(
        requirements=req,
        manifest=kb_dir / "knowledge_manifest.json",
        extra_topics=["Leave", "Attendance", "Rules", "Salary", "Promotions"],
        use_llm=False,
    )
    # Start with intentionally constrained top_k=1 so optimizer diagnoses retrieval_miss & prompt_defect
    agent = AgentGenerator().generate(
        requirements=req,
        supported_topic_config=topic_cfg,
        knowledge_manifest=kb_dir / "knowledge_manifest.json",
        top_k=1,
        use_llm=False,
    )
    dataset = EvalDatasetGenerator().generate(agent_def=agent, manifest=manifest, use_llm=False)

    opt_out = tmp_path / "opt"
    best_agent, log, init_rep, final_rep = CriticOptimizerLoop().optimize(
        agent_def=agent,
        eval_dataset=dataset,
        max_iterations=10,  # must hard-cap to 3
        use_llm=False,
        output_dir=opt_out,
    )

    assert log.max_iterations == MAX_OPTIMIZATION_ITERATIONS
    assert log.total_iterations_run <= MAX_OPTIMIZATION_ITERATIONS
    assert log.final_score >= log.initial_score
    assert (opt_out / "iteration_log.json").is_file()
    saved_log = json.loads((opt_out / "iteration_log.json").read_text(encoding="utf-8"))
    if saved_log["iterations"]:
        first_iter = saved_log["iterations"][0]
        assert {"iteration", "failure_type", "reason", "change_applied", "tests_addressed", "score_before", "score_after"} <= set(
            first_iter.keys()
        )
    assert best_agent.validation.is_valid is True


def test_critic_optimizer_rolls_back_on_regression(full_hr_docs: tuple[Path, Path], tmp_path: Path):
    brief, docs = full_hr_docs
    req = IntakeInterviewer().run(brief=brief, docs_dir=docs, config={"use_llm": False})
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    agent = AgentGenerator().generate(
        requirements=req,
        knowledge_manifest=kb_dir / "knowledge_manifest.json",
        use_llm=False,
    )
    dataset = EvalDatasetGenerator().generate(agent_def=agent, manifest=manifest, use_llm=False)

    # Custom factory: initial agent passes 17/20, mutated agent fails all 20 -> triggers rollback!
    def factory(cand_agent):
        if cand_agent.retrieval_config.top_k > agent.retrieval_config.top_k or "detail_mode" in cand_agent.system_prompt:
            return lambda q: AgentResponse(answer="Bad empty response", sources=[], confidence="low")
        return lambda q: AgentResponse(
            answer="Full-time employees receive 20 days annual leave, 10 sick days, core hours 10:00 AM to 4:00 PM.",
            sources=[d.document_name for d in manifest.documents],
            confidence="high",
            retrieved_context=[c.content for c in manifest.chunks],
        )

    best_agent, log, init_rep, final_rep = CriticOptimizerLoop().optimize(
        agent_def=agent,
        eval_dataset=dataset,
        max_iterations=2,
        agent_callable_factory=factory,
    )
    assert any(it.rolled_back for it in log.iterations)
    assert final_rep.scorecard.pass_rate == init_rep.scorecard.pass_rate
    assert best_agent.retrieval_config.top_k == agent.retrieval_config.top_k


def test_failure_analyzer_classifies_all_failure_types(full_hr_docs: tuple[Path, Path], tmp_path: Path):
    brief, docs = full_hr_docs
    req = IntakeInterviewer().run(brief=brief, docs_dir=docs, config={"use_llm": False})
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    agent = AgentGenerator().generate(
        requirements=req,
        knowledge_manifest=kb_dir / "knowledge_manifest.json",
        use_llm=False,
    )
    dataset = EvalDatasetGenerator().generate(agent_def=agent, manifest=manifest, use_llm=False)

    judge = EvaluationJudge()
    cases = dataset.cases
    fake_results = [
        # output_failure
        judge.evaluate_single(cases[0], AgentResponse(answer="   ", sources=[], confidence="high")),
        # wrong_refusal
        judge.evaluate_single(
            cases[1],
            AgentResponse(answer="I cannot answer this.", sources=[], confidence="unsupported", refused_unsupported_topic=True),
        ),
        # hallucination
        judge.evaluate_single(
            cases[2],
            AgentResponse(
                answer="Employees receive 999 days of leave.",
                sources=cases[2].expected_sources,
                confidence="high",
                retrieved_context=[f"[{cases[2].expected_sources[0]}] Employees receive 20 days of annual leave."],
            ),
        ),
        # retrieval_miss
        judge.evaluate_single(
            cases[3],
            AgentResponse(
                answer="Something irrelevant",
                sources=[],
                confidence="low",
                retrieved_context=[],
            ),
        ),
    ]
    from forgeai.eval.schemas import EvaluationReport, EvaluationScorecard

    dummy_report = EvaluationReport(
        evaluation_mode="test",
        scorecard=EvaluationScorecard(
            agent_name=agent.agent_name,
            total_cases=len(fake_results),
            passed_cases=0,
            failed_cases=len(fake_results),
            pass_rate=0.0,
            groundedness_rate=0.0,
            correctness_rate=0.0,
            refusal_accuracy_rate=0.0,
            source_quality_rate=0.0,
            contract_compliance_rate=0.0,
        ),
        results=fake_results,
    )
    diagnoses = FailureAnalyzer().analyze(dummy_report, dataset, agent)
    types = {d.failure_type for d in diagnoses}
    assert {"output_failure", "wrong_refusal", "hallucination", "retrieval_miss"} <= types


def test_runtime_chat_memory_tools_citations_and_refusal(full_hr_docs: tuple[Path, Path], tmp_path: Path):
    brief, docs = full_hr_docs
    req = IntakeInterviewer().run(brief=brief, docs_dir=docs, config={"use_llm": False})
    kb_dir = tmp_path / "kb"
    KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    _, topic_cfg = GroundabilityGuard().evaluate(
        requirements=req,
        manifest=kb_dir / "knowledge_manifest.json",
        extra_topics=["Leave", "Attendance", "Salary"],
        use_llm=False,
    )
    agent_path = tmp_path / "agent.json"
    AgentGenerator().generate(
        requirements=req,
        supported_topic_config=topic_cfg,
        knowledge_manifest=kb_dir / "knowledge_manifest.json",
        use_llm=False,
        output_path=agent_path,
    )

    runtime = AgentRuntime.from_file(agent_path)

    # Turn 1: Grounded policy question + citation
    r1 = runtime.chat("How many days of annual paid leave do employees receive?", use_llm=False)
    assert "20 days" in r1.answer
    assert "leave_policy.txt" in r1.sources
    assert "Source:" in r1.answer

    # Turn 2: Calculator tool invocation
    r2 = runtime.chat("What is 20 + 10 leave days?", use_llm=False)
    assert any(tc.tool_name == "calculator" and tc.output_preview == "30" for tc in r2.tool_calls)

    # Turn 3: Unsupported topic refusal
    r3 = runtime.chat("What is my current monthly salary?", use_llm=False)
    assert r3.refused_unsupported_topic is True
    assert r3.sources == []
    assert r3.confidence == "unsupported"

    # Memory persistence check
    assert len(runtime.memory.get_history()) == 6
    hist_path = tmp_path / "history.json"
    runtime.memory.save(hist_path)
    reloaded = ConversationMemory()
    reloaded.load(hist_path)
    assert len(reloaded.get_history()) == 6

    # Runtime CLI check
    assert runtime_cli_main(["--agent", str(agent_path), "-m", "What are core hours?", "--no-llm"]) == 0


def test_full_10_feature_pipeline_end_to_end(full_hr_docs: tuple[Path, Path], tmp_path: Path):
    brief, docs = full_hr_docs
    out_dir = tmp_path / "forgeai_run"

    result = ForgeAIPipeline().run(
        brief=brief,
        docs_dir=docs,
        output_dir=out_dir,
        extra_topics=["Leave", "Attendance", "Rules", "Salary", "Promotions"],
        use_llm=False,
    )

    assert (out_dir / "requirements.json").is_file()
    assert (out_dir / "kb" / "knowledge_manifest.json").is_file()
    assert (out_dir / "groundability" / "groundability_report.md").is_file()
    assert (out_dir / "tools" / "tool_selection.json").is_file()
    assert (out_dir / "agent.json").is_file()
    assert (out_dir / "eval" / "evals.json").is_file()
    assert (out_dir / "eval" / "iteration_log.json").is_file()
    assert (out_dir / "final_report.md").is_file()

    report_md = (out_dir / "final_report.md").read_text(encoding="utf-8")
    for required_section in (
        "FORGEAI QUALITY REPORT",
        "1. Input Brief",
        "2. Documents Analyzed",
        "3. Topic Scope & Groundability",
        "4. Selected Tools",
        "5. Retrieval Configuration",
        "6. Generated System Prompt",
        "7. Evaluation & Optimization Trajectory",
        "8. Failed Test Cases (Honest Disclosure)",
        "9. Remaining Weaknesses",
        "10. Recommendations",
    ):
        assert required_section in report_md

    # Working runtime returned from pipeline
    reply = result.runtime.chat("How many sick days do full-time employees get?", use_llm=False)
    assert "10 paid sick days" in reply.answer

