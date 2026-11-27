"""Comprehensive tests for Features 6 & 7 — Automatic Eval Dataset Generator & Hybrid Judge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from forgeai.eval import (
    MIN_EVAL_CASES,
    AgentResponse,
    EvalCase,
    EvalDataset,
    EvalDatasetGenerator,
    EvaluationJudge,
    EvaluationReport,
)
from forgeai.eval.cli import main as eval_cli_main
from forgeai.eval.schemas import EvalCaseBatch, LLMJudgeVerdict
from forgeai.generator import AgentGenerator
from forgeai.groundability import GroundabilityGuard
from forgeai.intake import IntakeInterviewer
from forgeai.knowledge import KnowledgeProcessor


@pytest.fixture()
def complete_hr_workspace(tmp_path: Path) -> tuple[Path, Path]:
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
    (docs / "benefits_overview.md").write_text(
        "# Benefits\n## Health Insurance\nMedical and dental insurance begin on the first day after hire.\n"
        "## Wellness Stipend\nEmployees receive a 500 USD annual wellness stipend.\n",
        encoding="utf-8",
    )

    req_path = tmp_path / "requirements.json"
    IntakeInterviewer().run(
        brief="Build an HR assistant for employees about leave policies, attendance, and benefits.",
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )
    kb_dir = tmp_path / "kb"
    KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    manifest_path = kb_dir / "knowledge_manifest.json"

    ground_dir = tmp_path / "groundability"
    GroundabilityGuard().evaluate(
        requirements=req_path,
        manifest=manifest_path,
        extra_topics=["Leave", "Attendance", "Benefits", "Salary", "Promotions"],
        use_llm=False,
        output_dir=ground_dir,
    )
    agent_path = tmp_path / "agent.json"
    AgentGenerator().generate(
        requirements=req_path,
        supported_topic_config=ground_dir / "supported_topic_config.json",
        knowledge_manifest=manifest_path,
        use_llm=False,
        output_path=agent_path,
    )
    return agent_path, manifest_path


def test_eval_dataset_generator_creates_at_least_20_non_hardcoded_cases(
    complete_hr_workspace: tuple[Path, Path], tmp_path: Path
):
    agent_path, manifest_path = complete_hr_workspace
    evals_path = tmp_path / "evals.json"

    dataset = EvalDatasetGenerator().generate(
        agent_def=agent_path,
        manifest=manifest_path,
        use_llm=False,
        output_path=evals_path,
    )

    assert evals_path.is_file()
    assert dataset.total_cases >= MIN_EVAL_CASES
    assert dataset.category_counts == {
        "answerable": 8,
        "multi_document": 4,
        "edge_case": 3,
        "ambiguous": 2,
        "unsupported": 3,
    }

    # Verify duplicate detection (all normalized questions are unique)
    questions_norm = [c.question.lower().strip() for c in dataset.cases]
    assert len(questions_norm) == len(set(questions_norm))

    # Verify unsupported cases require refusal and cite no sources
    unsupported = [c for c in dataset.cases if c.category == "unsupported"]
    assert all(c.should_refuse is True and c.expected_sources == [] for c in unsupported)

    # Verify multi-document cases cite >= 2 distinct documents
    multi_doc = [c for c in dataset.cases if c.category == "multi_document"]
    assert all(len(set(c.expected_sources)) >= 2 for c in multi_doc)


def test_eval_dataset_schema_rejects_under_20_or_duplicate_ids():
    too_few = [
        EvalCase(test_id=f"T_{i}", category="answerable", question=f"Question number {i}?")
        for i in range(5)
    ]
    with pytest.raises(ValidationError):
        EvalDataset(agent_name="Test", total_cases=5, cases=too_few)

    dup_ids = [
        EvalCase(test_id="DUP_1", category="answerable", question=f"Unique question {i}?")
        for i in range(20)
    ]
    with pytest.raises(ValidationError, match="Duplicate test_id"):
        EvalDataset(agent_name="Test", total_cases=20, cases=dup_ids)


def test_eval_judge_executes_generated_agent_end_to_end(
    complete_hr_workspace: tuple[Path, Path], tmp_path: Path
):
    agent_path, manifest_path = complete_hr_workspace
    dataset = EvalDatasetGenerator().generate(
        agent_def=agent_path,
        manifest=manifest_path,
        use_llm=False,
    )

    eval_out = tmp_path / "eval_out"
    report = EvaluationJudge().evaluate(
        agent_def=agent_path,
        eval_dataset=dataset,
        use_llm=False,
        output_dir=eval_out,
    )

    assert (eval_out / "eval_results.json").is_file()
    assert (eval_out / "scorecard.md").is_file()
    assert report.scorecard.total_cases == 20
    assert report.scorecard.refusal_accuracy_rate == 1.0
    assert report.scorecard.contract_compliance_rate == 1.0
    assert report.scorecard.pass_rate >= 0.75
    assert "EVALUATION SCORECARD" in report.formatted_scorecard


def test_eval_judge_catches_hallucinations_wrong_facts_and_missed_refusal():
    judge = EvaluationJudge()

    # 1. Ungrounded / hallucinated number -> Fail
    c1 = EvalCase(
        test_id="HR_001",
        category="answerable",
        question="How many annual leave days are provided?",
        expected_answer="Employees receive 20 days of annual paid leave.",
        expected_sources=["leave_policy.txt"],
        expected_facts=["20 days", "annual"],
        should_refuse=False,
    )
    bad_resp = AgentResponse(
        answer="Employees receive 99 days of annual paid leave.",
        sources=["leave_policy.txt"],
        confidence="high",
        retrieved_context=["[leave_policy.txt] Full-time employees receive 20 days of annual paid leave."],
        raw_json={"answer": "Employees receive 99 days", "sources": ["leave_policy.txt"], "confidence": "high"},
    )
    r1 = judge.evaluate_single(c1, bad_resp, allowed_sources=["leave_policy.txt"])
    assert r1.grounded == "No"
    assert r1.correct == "No"
    assert r1.score_label == "Fail"

    # 2. Failed refusal on unsupported topic -> Fail
    c2 = EvalCase(
        test_id="HR_019",
        category="unsupported",
        question="What is my current salary?",
        expected_sources=[],
        should_refuse=True,
    )
    hallucinated_salary = AgentResponse(
        answer="Your current monthly salary is 8000 USD.",
        sources=[],
        confidence="high",
        raw_json={"answer": "Your current monthly salary is 8000 USD.", "sources": [], "confidence": "high"},
    )
    r2 = judge.evaluate_single(c2, hallucinated_salary)
    assert r2.refusal == "No"
    assert r2.score_label == "Fail"

    # 3. Proper refusal -> Pass with N/A for Grounded & Correct
    good_refusal = AgentResponse(
        answer="I cannot answer questions about salary because no payroll data is available.",
        sources=[],
        confidence="unsupported",
        refused_unsupported_topic=True,
        raw_json={"answer": "I cannot answer", "sources": [], "confidence": "unsupported"},
    )
    r3 = judge.evaluate_single(c2, good_refusal)
    assert r3.grounded == "N/A"
    assert r3.correct == "N/A"
    assert r3.refusal == "Yes"
    assert r3.score_label == "Pass"


def test_eval_judge_llm_mode_mocked(complete_hr_workspace: tuple[Path, Path]):
    agent_path, manifest_path = complete_hr_workspace
    dataset = EvalDatasetGenerator().generate(
        agent_def=agent_path,
        manifest=manifest_path,
        use_llm=False,
    )

    with (
        patch("forgeai.eval.agent_runner.complete_structured") as mock_agent_llm,
        patch("forgeai.eval.judge.complete_structured") as mock_judge_llm,
    ):
        from forgeai.eval.agent_runner import _LLMAgentOutput

        mock_agent_llm.return_value = _LLMAgentOutput(
            answer="Full-time employees receive 20 days of annual leave and core hours are 10:00 AM to 4:00 PM.",
            sources=["leave_policy.txt", "attendance_rules.csv"],
            confidence="high",
            refused_unsupported_topic=False,
        )
        mock_judge_llm.return_value = LLMJudgeVerdict(
            grounded=True,
            correct=True,
            refusal_appropriate=True,
            reasoning="Grounded in retrieved context.",
        )

        # Evaluate subset of 2 cases cleanly via dataset copy? No, EvalDataset enforces >=20; test 1 single call:
        r = EvaluationJudge().evaluate_single(
            case=dataset.cases[0],
            response=AgentResponse(
                answer="Full-time employees receive 20 days of annual leave.",
                sources=["leave_policy.txt"],
                confidence="high",
                retrieved_context=["Full-time employees receive 20 days of annual paid leave."],
            ),
            allowed_sources=["leave_policy.txt", "attendance_rules.csv", "benefits_overview.md"],
            use_llm_judge=True,
        )
        mock_judge_llm.assert_called_once()
        assert r.llm_judge_notes == "Grounded in retrieved context."


def test_eval_cli_run_all(complete_hr_workspace: tuple[Path, Path], tmp_path: Path):
    agent_path, manifest_path = complete_hr_workspace
    out_dir = tmp_path / "cli_eval"
    code = eval_cli_main(
        [
            "run-all",
            "--agent",
            str(agent_path),
            "--manifest",
            str(manifest_path),
            "--no-llm",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "evals.json").is_file()
    assert (out_dir / "eval_results.json").is_file()
    assert (out_dir / "scorecard.md").is_file()
    report = EvaluationReport.model_validate_json((out_dir / "eval_results.json").read_text(encoding="utf-8"))
    assert report.scorecard.total_cases >= 20

