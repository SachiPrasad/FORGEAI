"""Comprehensive tests for Feature 4 — Agent Generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from forgeai.generator import (
    AgentDefinition,
    AgentDefinitionValidator,
    AgentGenerator,
    PromptGenerationEngine,
    RetrievalConfig,
    ToolSelector,
)
from forgeai.generator.cli import main as generator_cli_main
from forgeai.generator.prompt_engine import _PolishedPrompt
from forgeai.groundability import GroundabilityGuard
from forgeai.intake import IntakeInterviewer
from forgeai.knowledge import KnowledgeProcessor


@pytest.fixture()
def full_pipeline_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave_policy.txt").write_text(
        "Annual Leave:\nEmployees receive 20 days of annual paid leave and 10 sick days per year.",
        encoding="utf-8",
    )
    (docs / "attendance_policy.txt").write_text(
        "Attendance Rules:\nCore hours are 10:00 AM to 4:00 PM Monday through Friday.",
        encoding="utf-8",
    )

    req_path = tmp_path / "requirements.json"
    IntakeInterviewer().run(
        brief="Build an HR assistant for employees that answers questions about leave policies and attendance.",
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
        extra_topics=["Leave", "Attendance", "Salary"],
        use_llm=False,
        output_dir=ground_dir,
    )
    return req_path, ground_dir / "supported_topic_config.json", manifest_path


def test_agent_generator_produces_valid_agent_json(full_pipeline_inputs: tuple[Path, Path, Path], tmp_path: Path):
    req_path, ground_path, manifest_path = full_pipeline_inputs
    out_file = tmp_path / "agent.json"

    agent_def = AgentGenerator().generate(
        requirements=req_path,
        supported_topic_config=ground_path,
        knowledge_manifest=manifest_path,
        model="openai/o3",
        chunk_size_override=700,
        chunk_overlap_override=100,
        top_k=4,
        use_llm=False,
        output_path=out_file,
    )

    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    # Verify all required top-level keys from Feature 4 spec
    for key in (
        "agent_name",
        "model",
        "model_config",
        "system_prompt",
        "retrieval_config",
        "tools",
        "supported_topics",
        "unsupported_topics",
        "output_contract",
    ):
        assert key in payload

    assert payload["agent_name"] == "HR Policy Assistant"
    assert payload["model"] == "openai/o3"
    assert payload["retrieval_config"]["chunk_size"] == 700
    assert payload["retrieval_config"]["chunk_overlap"] == 100
    assert payload["retrieval_config"]["top_k"] == 4
    assert payload["output_contract"]["answer"] == "string"
    assert payload["output_contract"]["sources"] == []
    assert payload["output_contract"]["confidence"] == "string"

    tool_names = [t["name"] for t in payload["tools"]]
    assert "document_retriever" in tool_names
    assert "verify_topic_scope" in tool_names
    assert "web_search" not in tool_names
    assert payload["tool_selection_record"]["selected_tools"]

    assert "Never invent company policies" in payload["system_prompt"]
    assert "Salary" in payload["unsupported_topics"]
    assert agent_def.validation.is_valid is True
    assert "class StructuredOutput(BaseModel):" in agent_def.structured_output_code


def test_tool_selector_enables_nlp_and_blocks_web_under_kb_restrictions(
    full_pipeline_inputs: tuple[Path, Path, Path],
):
    req_path, ground_path, manifest_path = full_pipeline_inputs
    from forgeai.groundability.schemas import SupportedTopicConfig
    from forgeai.intake.schemas import AgentRequirements
    from forgeai.knowledge.schemas import KnowledgeManifest

    req_obj = AgentRequirements.model_validate_json(req_path.read_text(encoding="utf-8"))
    req_obj = req_obj.model_copy(update={"purpose": "Summarize HR policies and browse the web"})
    topic_cfg = SupportedTopicConfig.model_validate_json(ground_path.read_text(encoding="utf-8"))
    manifest_obj = KnowledgeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    tools, rejected = ToolSelector().select(
        requirements=req_obj,
        topic_config=topic_cfg,
        retrieval_config=RetrievalConfig(knowledge_manifest_path=str(manifest_path)),
        manifest=manifest_obj,
    )
    selected_names = {t.name for t in tools}
    assert "summarize_text_with_llm" in selected_names
    assert "web_search" not in selected_names
    assert any(r["name"] == "web_search" for r in rejected)


def test_prompt_engine_llm_safety_anchor_fallback(full_pipeline_inputs: tuple[Path, Path, Path]):
    req_path, ground_path, _ = full_pipeline_inputs
    from forgeai.groundability.schemas import SupportedTopicConfig
    from forgeai.intake.schemas import AgentRequirements

    req_obj = AgentRequirements.model_validate_json(req_path.read_text(encoding="utf-8"))
    topic_cfg = SupportedTopicConfig.model_validate_json(ground_path.read_text(encoding="utf-8"))

    # If an LLM drops critical safety/refusal anchors, engine falls back to deterministic safe prompt
    unsafe_polished = _PolishedPrompt(system_prompt="You are a helpful HR bot. Answer anything the user asks warmly.")
    with patch("forgeai.generator.prompt_engine.complete_structured", return_value=unsafe_polished):
        prompt = PromptGenerationEngine().generate(req_obj, topic_cfg, tools=[], use_llm=True)
        assert "Never invent company policies" in prompt
        assert "Salary" in prompt


def test_validator_detects_missing_retrieval_and_unsafe_prompt(full_pipeline_inputs: tuple[Path, Path, Path]):
    req_path, ground_path, manifest_path = full_pipeline_inputs
    agent_def = AgentGenerator().generate(
        requirements=req_path,
        supported_topic_config=ground_path,
        knowledge_manifest=manifest_path,
        use_llm=False,
    )
    broken = agent_def.model_copy(
        update={
            "tools": [],
            "system_prompt": "Short prompt without any rules or refusal wording.",
            "output_contract": {"answer": "string"},
        }
    )
    report = AgentDefinitionValidator().validate(broken, has_knowledge_base=True)
    assert not report.is_valid
    components = {i.component for i in report.issues}
    assert {"system_prompt", "tools", "output_contract"} <= components


def test_generator_cli_writes_agent_json(full_pipeline_inputs: tuple[Path, Path, Path], tmp_path: Path):
    req_path, ground_path, manifest_path = full_pipeline_inputs
    out_file = tmp_path / "cli_agent.json"
    code = generator_cli_main(
        [
            "--requirements",
            str(req_path),
            "--groundability",
            str(ground_path),
            "--manifest",
            str(manifest_path),
            "--model",
            "gpt-4o-mini",
            "--no-llm",
            "--out",
            str(out_file),
        ]
    )
    assert code == 0
    loaded = AgentDefinition.model_validate_json(out_file.read_text(encoding="utf-8"))
    assert loaded.validation.is_valid is True
