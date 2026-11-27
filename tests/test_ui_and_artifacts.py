"""Tests for the 7-tab UI dashboard, Streamlit AppTest, scorecards.json, and canonical artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.knowledge.retriever import KnowledgeRetriever
from forgeai.pipeline import ForgeAIPipeline
from forgeai.ui import load_workspace_artifacts


def test_canonical_section8_artifacts_and_scorecards(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.txt").write_text(
        "Annual Leave:\nEmployees receive 20 days of annual paid leave and 10 sick days per year.",
        encoding="utf-8",
    )
    (docs / "attendance.txt").write_text(
        "Core Hours:\nCore hours are 10:00 AM to 4:00 PM Monday through Friday.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    res = ForgeAIPipeline().run(
        brief="Build an HR assistant for employees about leave policies and attendance.",
        docs_dir=docs,
        output_dir=out_dir,
        use_llm=False,
    )

    # Verify all 8 Section-8 canonical artifact files exist at out_dir root
    for filename in (
        "requirements.json",
        "groundability_report.json",
        "agent.json",
        "knowledge_manifest.json",
        "evals.json",
        "scorecards.json",
        "iteration_log.json",
        "final_report.md",
        "dashboard.html",
        "trace_log.json",
    ):
        assert (out_dir / filename).is_file(), f"Missing canonical artifact: {filename}"

    assert (out_dir / "logs" / "forgeai.log").is_file()

    # Verify root knowledge_manifest.json works directly with KnowledgeRetriever
    hits = KnowledgeRetriever.from_manifest(out_dir / "knowledge_manifest.json").retrieve("annual leave", top_k=2)
    assert hits

    # Verify honest eval pass rate terminology & benchmark scope note (Change A)
    report_md = (out_dir / "final_report.md").read_text(encoding="utf-8")
    assert "Benchmark Scope Note:" in report_md
    assert "INITIAL EVALUATION PASS RATE:" in report_md
    assert "FINAL EVALUATION PASS RATE:" in report_md

    # Verify version_diff.json includes config_diff and regression_count (Change D)
    diff_data = json.loads((out_dir / "eval" / "versions" / "version_diff.json").read_text(encoding="utf-8"))
    assert "config_diff" in diff_data
    assert "regression_count" in diff_data

    # Verify scorecards.json contains Initial (iteration 0) + iteration scorecards
    sc_data = json.loads((out_dir / "scorecards.json").read_text(encoding="utf-8"))
    assert sc_data["scorecards"][0]["iteration"] == 0
    assert sc_data["initial_score"] == res.optimization_log.initial_score

    # Verify 7-tab HTML dashboard content
    html_text = (out_dir / "dashboard.html").read_text(encoding="utf-8")
    for tab_label in (
        "Project Intake",
        "Knowledge Base",
        "Agent Configuration",
        "Evaluation",
        "Optimization",
        "Final Report",
        "Test Your Agent",
        "What Changed",
    ):
        assert tab_label in html_text

    # Verify UI artifact loader
    loaded = load_workspace_artifacts(out_dir)
    assert loaded["agent"].agent_name == res.agent.agent_name
    assert loaded["scorecards"] is not None
    assert loaded["version_diff"] is not None


def test_runtime_live_llm_fallback_and_multi_part_refusal(tmp_path: Path):
    from unittest.mock import patch

    from forgeai.runtime import AgentRuntime

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.txt").write_text("Annual Leave:\nEmployees receive 20 days of annual leave.", encoding="utf-8")
    out_dir = tmp_path / "run"
    res = ForgeAIPipeline().run(
        brief="Build an HR assistant for employees about leave policies.",
        docs_dir=docs,
        extra_topics=["Salary", "Promotions"],
        output_dir=out_dir,
        use_llm=False,
    )
    runtime = AgentRuntime(res.agent)

    # Change C: Failure / Refusal demo
    refusal = runtime.chat("What is my monthly salary, and when will I get promoted?", use_llm=False)
    assert refusal.refused_unsupported_topic is True
    assert refusal.sources == []
    assert "Sources: No supporting document found." in refusal.answer

    # Change B: Live LLM failure falls back gracefully to deterministic_fallback
    with patch("litellm.completion", side_effect=RuntimeError("API unreachable")):
        fallback = runtime.chat("How many days of annual leave do employees receive?", use_llm=True)
        assert fallback.mode_used == "deterministic_fallback"
        assert "20 days" in fallback.answer


def test_streamlit_app_renders_cleanly():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 7
    assert len(at.metric) >= 4


def test_portable_runtime_paths_and_model_resolution(tmp_path: Path, monkeypatch):
    from forgeai.intake.llm_client import resolve_effective_model
    from forgeai.runtime import AgentRuntime

    # 1. Portable resolution when agent.json has a non-existent Windows path
    demo_out = Path("demos/demo_a_hr_assistant/output")
    if demo_out.is_dir():
        from forgeai.ui import load_workspace_artifacts

        loaded = load_workspace_artifacts(demo_out)
        broken_agent = loaded["agent"].model_copy(
            update={
                "retrieval_config": loaded["agent"].retrieval_config.model_copy(
                    update={"knowledge_manifest_path": "C:\\stale\\path\\kb\\knowledge_manifest.json"}
                )
            }
        )
        rt = AgentRuntime(broken_agent, workspace_dir=demo_out)
        assert rt.retriever is not None
        reply = rt.chat("How many days of annual leave do employees get?", use_llm=False)
        assert reply.retrieved_chunk_ids

    # 2. Provider model fallback when only GEMINI_API_KEY is configured
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    assert resolve_effective_model("gpt-4o-mini") == "gemini/gemini-2.0-flash"

