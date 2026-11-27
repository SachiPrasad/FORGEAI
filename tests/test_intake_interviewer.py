"""Comprehensive tests for Feature 1 — Intelligent Intake Interviewer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from docx import Document

from forgeai.intake import (
    MAX_CLARIFICATION_QUESTIONS,
    AgentRequirements,
    BriefParser,
    ClarificationGenerator,
    ClarificationQuestion,
    DocumentAnalyzer,
    IntakeInterviewer,
    RequirementValidator,
    UserConfig,
)
from forgeai.intake.cli import main as cli_main
from forgeai.intake.schemas import ClarificationBatch, ExtractedRequirementsDraft


def _write_minimal_pdf(path: Path, text_lines: list[str]) -> None:
    """Create a valid single-page PDF with extractable Helvetica text."""
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in text_lines]
    stream_cmds = ["BT", "/F1 12 Tf", "72 720 Td"]
    for idx, line in enumerate(escaped):
        if idx > 0:
            stream_cmds.append("0 -18 Td")
        stream_cmds.append(f"({line}) Tj")
    stream_cmds.append("ET")
    stream_data = "\n".join(stream_cmds).encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
        + stream_data
        + b"\nendstream\nendobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    path.write_bytes(bytes(pdf))


@pytest.fixture()
def sample_hr_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "hr_handbook.txt").write_text(
        "# HR Handbook\nEmployees must follow workplace conduct and standard office hours.",
        encoding="utf-8",
    )

    (docs / "attendance_rules.csv").write_text(
        "rule_id,category,rule_summary\nATT-01,Core Hours,10am to 4pm\n",
        encoding="utf-8",
    )

    docx_doc = Document()
    docx_doc.add_heading("Leave Policy", level=1)
    docx_doc.add_paragraph("Full-time employees receive 20 days of annual paid leave and 10 sick days.")
    docx_doc.save(str(docs / "leave_policy.docx"))

    _write_minimal_pdf(
        docs / "benefits_overview.pdf",
        ["Benefits Overview", "Health insurance, dental coverage, and wellness stipend are included."],
    )

    # Unsupported file should be gracefully skipped
    (docs / "ignore_me.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return docs


def test_brief_parser_extracts_hr_signals():
    brief = (
        "Build an HR assistant for employees that can answer questions about "
        "leave policies, attendance, benefits and company rules."
    )
    parsed = BriefParser().parse(brief)
    assert "Human Resources" in parsed.domain_hints
    assert "Employees" in parsed.audience_hints
    assert "leave policies" in parsed.candidate_topics
    assert "attendance" in parsed.candidate_topics
    assert "benefits" in parsed.candidate_topics
    assert "unclear_policy_only_vs_balance_calculation" in parsed.ambiguity_signals


def test_brief_parser_rejects_empty_brief():
    with pytest.raises(ValueError, match="empty"):
        BriefParser().parse("   \n  ")


def test_document_analyzer_reads_pdf_docx_txt_csv(sample_hr_docs: Path):
    result = DocumentAnalyzer().analyze_folder(sample_hr_docs)
    assert result.supported_files_count == 4
    assert result.skipped_files == ["ignore_me.png"]
    by_type = {d.doc_type: d for d in result.documents}
    assert set(by_type.keys()) == {"pdf", "docx", "txt", "csv"}
    assert "20 days of annual paid leave" in by_type["docx"].full_text
    assert "Health insurance" in by_type["pdf"].full_text
    assert by_type["csv"].columns == ["rule_id", "category", "rule_summary"]
    assert by_type["csv"].row_count == 1


def test_clarifier_respects_five_question_cap(sample_hr_docs: Path):
    parsed = BriefParser().parse(
        "Build an HR assistant to answer questions about leave policies and submit requests."
    )
    doc_analysis = DocumentAnalyzer().analyze_folder(sample_hr_docs)
    draft = ExtractedRequirementsDraft(
        agent_name="HR Policy Assistant",
        domain="Human Resources",
        target_audience="Employees",
        purpose="Answer HR policy questions",
        supported_topics=["Leave", "Benefits", "Attendance"],
        unsupported_topics=["Payroll calculations without payroll data"],
        tone="Professional and friendly",
        output_format="Conversational answers",
        restrictions=["No invented HR policy"],
        sources=doc_analysis.available_sources,
    )
    cfg = UserConfig(max_clarification_questions=20)  # should cap at 5
    assert cfg.max_clarification_questions == MAX_CLARIFICATION_QUESTIONS

    questions = ClarificationGenerator().generate(parsed, doc_analysis, draft, cfg, use_llm=False)
    assert 1 <= len(questions) <= MAX_CLARIFICATION_QUESTIONS
    assert any("leave balance" in q.question.lower() for q in questions)


def test_validator_catches_overlap_and_question_overflow(sample_hr_docs: Path):
    from pydantic import ValidationError

    doc_analysis = DocumentAnalyzer().analyze_folder(sample_hr_docs)
    too_many_questions = [
        ClarificationQuestion(id=f"q{i}", question=f"Question number {i}?", reason="test")
        for i in range(1, 8)
    ]

    with pytest.raises(ValidationError):
        AgentRequirements(
            agent_name="HR Policy Assistant",
            domain="Human Resources",
            target_audience="Employees",
            purpose="Answer HR policy questions",
            supported_topics=["Leave", "Payroll"],
            unsupported_topics=["Payroll"],
            tone="Professional and friendly",
            output_format="Conversational answers",
            restrictions=["No invented HR policy"],
            sources=["HR handbook"],
            clarification_questions=too_many_questions,
        )

    unvalidated_req = AgentRequirements.model_construct(
        agent_name="HR Policy Assistant",
        domain="Human Resources",
        target_audience="Employees",
        purpose="Answer HR policy questions",
        supported_topics=["Leave", "Payroll"],
        unsupported_topics=["Payroll"],
        tone="Professional and friendly",
        output_format="Conversational answers",
        restrictions=["No invented HR policy"],
        sources=["HR handbook"],
        clarification_questions=too_many_questions,
    )
    report = RequirementValidator().validate(unvalidated_req, doc_analysis=doc_analysis)
    assert not report.is_valid
    assert any("both supported and unsupported" in issue.message for issue in report.issues)
    assert any("exceed the maximum allowed (5)" in issue.message for issue in report.issues)


def test_end_to_end_offline_intake_and_refinement(sample_hr_docs: Path, tmp_path: Path):
    output_file = tmp_path / "requirements.json"
    interviewer = IntakeInterviewer()
    brief = (
        "Build an HR assistant for employees that can answer questions about "
        "leave policies, attendance, benefits and company rules."
    )
    config = {
        "use_llm": False,
        "tone_override": "Professional and friendly",
        "output_format_override": "Conversational answers",
    }

    req = interviewer.run(
        brief=brief,
        docs_dir=sample_hr_docs,
        config=config,
        output_path=output_file,
    )

    assert output_file.is_file()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["agent_name"] == "HR Policy Assistant"
    assert saved["domain"] == "Human Resources"
    assert saved["target_audience"] == "Employees"
    assert saved["tone"] == "Professional and friendly"
    assert saved["output_format"] == "Conversational answers"
    assert "Payroll calculations without payroll data" in saved["unsupported_topics"]
    assert "No invented HR policy" in saved["restrictions"]
    assert len(saved["clarification_questions"]) <= MAX_CLARIFICATION_QUESTIONS
    assert saved["validation"]["is_valid"] is True

    # Test stakeholder clarification answer refinement
    first_q_id = req.clarification_questions[0].id
    refined = interviewer.refine_with_answers(
        requirements=req,
        answers={first_q_id: "Policy information only, do not calculate leave balances."},
        user_config=config,
        output_path=output_file,
    )
    assert refined.clarification_answers[first_q_id].startswith("Policy information only")
    assert any("do not compute personal leave balances" in r.lower() for r in refined.restrictions)


def test_end_to_end_llm_mode_mocked(sample_hr_docs: Path, tmp_path: Path):
    mock_draft = ExtractedRequirementsDraft(
        agent_name="HR Policy Assistant",
        domain="Human Resources",
        target_audience="Employees",
        purpose="Answer HR policy questions",
        supported_topics=["Leave", "Benefits", "Attendance"],
        unsupported_topics=["Payroll calculations without payroll data"],
        tone="Professional and friendly",
        output_format="Conversational answers",
        restrictions=["No invented HR policy"],
        sources=["HR handbook", "leave policy"],
    )
    mock_batch = ClarificationBatch(
        questions=[
            ClarificationQuestion(
                id="q1",
                question="Should this assistant provide policy information only, or should it also calculate an employee's leave balance?",
                reason="Leave balance data is not present in the document folder.",
                category="scope",
            )
        ]
    )

    with (
        patch("forgeai.intake.interviewer.complete_structured", return_value=mock_draft) as mock_extract,
        patch("forgeai.intake.clarifier.complete_structured", return_value=mock_batch) as mock_clarify,
    ):
        output_file = tmp_path / "llm_requirements.json"
        req = IntakeInterviewer().run(
            brief="Build an HR assistant for employees that can answer questions about leave policies, attendance, benefits and company rules.",
            docs_dir=sample_hr_docs,
            config={"use_llm": True, "model": "openai/o3"},
            output_path=output_file,
        )
        mock_extract.assert_called_once()
        mock_clarify.assert_called_once()
        assert req.metadata["extraction_mode"] == "llm"
        assert req.agent_name == "HR Policy Assistant"
        assert req.validation.is_valid is True


def test_cli_non_interactive_outputs_requirements_json(sample_hr_docs: Path, tmp_path: Path):
    brief_path = tmp_path / "brief.txt"
    brief_path.write_text(
        "Build an HR assistant for employees that can answer questions about leave policies, attendance, benefits and company rules.",
        encoding="utf-8",
    )
    out_path = tmp_path / "cli_requirements.json"
    exit_code = cli_main(
        [
            "--brief",
            str(brief_path),
            "--docs",
            str(sample_hr_docs),
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.is_file()
