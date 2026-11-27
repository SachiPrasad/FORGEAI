"""Comprehensive tests for Feature 3 — Groundability Guard."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from forgeai.groundability import GroundabilityGuard, GroundabilityReport, SupportedTopicConfig
from forgeai.groundability.cli import main as groundability_cli_main
from forgeai.groundability.schemas import LLMGroundabilityBatch, LLMTopicVerdict
from forgeai.intake import IntakeInterviewer
from forgeai.knowledge import KnowledgeProcessor


def _write_simple_pdf(path: Path, lines: list[str]) -> None:
    escaped = [l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for l in lines]
    cmds = ["BT", "/F1 12 Tf", "72 720 Td"]
    for idx, l in enumerate(escaped):
        if idx > 0:
            cmds.append("0 -18 Td")
        cmds.append(f"({l}) Tj")
    cmds.append("ET")
    stream = "\n".join(cmds).encode("latin-1")
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
        + stream
        + b"\nendstream\nendobj\n",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offs = [0]
    for obj in objects_iter(objs):
        offs.append(len(pdf))
        pdf.extend(obj)
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offs)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offs[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(offs)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(bytes(pdf))


def objects_iter(objs: list[bytes]) -> list[bytes]:
    return objs


@pytest.fixture()
def hr_scenario(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()

    _write_simple_pdf(
        docs / "Leave Policy.pdf",
        [
            "Leave Policy",
            "Full-time employees receive 20 days of annual paid leave and 10 sick days per year.",
            "Unused annual leave up to 5 days can carry over into the first quarter.",
        ],
    )
    _write_simple_pdf(
        docs / "Attendance Policy.pdf",
        [
            "Attendance Policy",
            "Core collaboration hours are 10:00 AM to 4:00 PM Monday through Friday.",
            "Employees must notify their manager before 10:00 AM if delayed.",
        ],
    )
    _write_simple_pdf(
        docs / "Employee Handbook.pdf",
        [
            "Employee Handbook",
            "All staff must follow workplace conduct and confidentiality standards.",
            "Basic medical insurance is offered to full-time employees; contact HR for plan details.",
        ],
    )

    brief = (
        "Create an HR assistant that answers questions about employee leave, salaries, "
        "promotions, medical insurance, and retirement benefits, plus attendance."
    )
    req_path = tmp_path / "requirements.json"
    IntakeInterviewer().run(
        brief=brief,
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )

    kb_dir = tmp_path / "kb"
    KnowledgeProcessor().process(
        docs_dir=docs,
        output_dir=kb_dir,
        embedding_provider="local",
    )
    return req_path, kb_dir / "knowledge_manifest.json"


def test_groundability_guard_classifies_supported_partial_unsupported(hr_scenario: tuple[Path, Path], tmp_path: Path):
    req_path, manifest_path = hr_scenario
    out_dir = tmp_path / "groundability"

    report, config = GroundabilityGuard().evaluate(
        requirements=req_path,
        manifest=manifest_path,
        extra_topics=[
            "Leave",
            "Attendance",
            "Salary",
            "Promotions",
            "Medical insurance",
            "Retirement benefits",
        ],
        use_llm=False,
        output_dir=out_dir,
    )

    by_topic = {a.topic.lower(): a for a in report.topic_assessments}
    assert by_topic["leave"].status == "supported"
    assert by_topic["leave"].document_support == "Available"
    assert by_topic["attendance"].status == "supported"
    assert by_topic["medical insurance"].status == "partial"
    assert by_topic["medical insurance"].document_support == "Partial"
    assert by_topic["salary"].status == "unsupported"
    assert by_topic["salary"].document_support == "Missing"
    assert by_topic["promotions"].status == "unsupported"
    assert by_topic["retirement benefits"].status == "unsupported"

    assert "GROUNDABILITY REPORT" in report.formatted_report
    assert "Generate the agent with supported topics only." in report.recommendation
    assert any("never invent" in r.lower() or "missing documents" in r.lower() for r in config.restrictions)
    assert "Salary" in config.refusal_policy
    assert (out_dir / "groundability_report.json").is_file()
    assert (out_dir / "groundability_report.md").is_file()
    assert (out_dir / "supported_topic_config.json").is_file()


def test_groundability_strict_no_partial_excludes_partial(hr_scenario: tuple[Path, Path]):
    req_path, manifest_path = hr_scenario
    report, config = GroundabilityGuard().evaluate(
        requirements=req_path,
        manifest=manifest_path,
        extra_topics=["Leave", "Medical insurance", "Salary"],
        allow_partial_topics=False,
        use_llm=False,
    )
    assert "Medical insurance" in report.partially_supported_topics
    assert config.partially_supported_topics == []
    assert "Medical insurance" in config.unsupported_topics


def test_groundability_all_unsupported_blocks_generation(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "office_wifi.txt").write_text("Guest Wi-Fi password rotates every Monday morning.", encoding="utf-8")

    req_path = tmp_path / "req.json"
    IntakeInterviewer().run(
        brief="Build an assistant for tax filing and stock options.",
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )
    kb_dir = tmp_path / "kb"
    KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")

    report, config = GroundabilityGuard().evaluate(
        requirements=req_path,
        manifest=kb_dir / "knowledge_manifest.json",
        extra_topics=["Tax filing", "Stock options"],
        use_llm=False,
    )
    assert report.supported_topics == []
    assert "Block agent generation" in report.recommendation
    assert report.groundability_score == 0.0


def test_groundability_llm_mode_mocked(hr_scenario: tuple[Path, Path]):
    req_path, manifest_path = hr_scenario
    mock_verdicts = LLMGroundabilityBatch(
        verdicts=[
            LLMTopicVerdict(topic="Leave", status="supported", reason="Backed by Leave Policy.pdf."),
            LLMTopicVerdict(topic="Salary", status="unsupported", reason="No salary records provided."),
        ]
    )
    with patch("forgeai.groundability.guard.complete_structured", return_value=mock_verdicts) as mock_llm:
        report, _ = GroundabilityGuard().evaluate(
            requirements=req_path,
            manifest=manifest_path,
            extra_topics=["Leave", "Salary"],
            use_llm=True,
        )
        mock_llm.assert_called_once()
        assert report.metadata["verification_mode"] == "llm_assisted"


def test_groundability_cli(hr_scenario: tuple[Path, Path], tmp_path: Path):
    req_path, manifest_path = hr_scenario
    out_dir = tmp_path / "cli_out"
    code = groundability_cli_main(
        [
            "--requirements",
            str(req_path),
            "--manifest",
            str(manifest_path),
            "--topics",
            "Leave",
            "Attendance",
            "Salary",
            "--no-llm",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0
    saved_report = GroundabilityReport.model_validate_json((out_dir / "groundability_report.json").read_text())
    saved_config = SupportedTopicConfig.model_validate_json((out_dir / "supported_topic_config.json").read_text())
    assert "Leave" in saved_report.supported_topics
    assert "Salary" in saved_config.unsupported_topics

