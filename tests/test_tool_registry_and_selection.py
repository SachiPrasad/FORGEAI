"""Comprehensive tests for Feature 5 — Tool Registry & Tool Selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from forgeai.groundability import GroundabilityGuard
from forgeai.intake import IntakeInterviewer
from forgeai.knowledge import KnowledgeProcessor
from forgeai.tools import (
    DynamicToolSelector,
    ToolRegistry,
    ToolRegistryEntry,
    ToolSelectionRecord,
    build_default_registry,
    calculator,
    csv_lookup,
    date_tool,
    http_api,
    web_search,
)
from forgeai.tools.cli import main as tools_cli_main
from forgeai.tools.schemas import LLMToolSelectionBatch, ToolDecision


def test_default_registry_contains_all_six_tools(tmp_path: Path):
    registry = build_default_registry()
    ids = {e.tool_id for e in registry.list_entries()}
    assert ids == {
        "document_retriever",
        "calculator",
        "csv_lookup",
        "date_tool",
        "http_api",
        "web_search",
    }
    export_path = registry.save_json(tmp_path / "tool_registry.json")
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert data["total_tools"] == 6


def test_tool_implementations_work_safely(tmp_path: Path):
    # Calculator (safe AST evaluation)
    assert calculator("20 * 5 + (12 / 4)") == "103"
    assert calculator("round(sqrt(16) + 2.5, 1)") == "6.5"
    assert "Error" in calculator("__import__('os').system('echo hi')")
    assert "Division by zero" in calculator("10 / 0")

    # CSV Lookup
    csv_file = tmp_path / "spare_parts_inventory.csv"
    csv_file.write_text(
        "part_id,name,stock,unit_price\n"
        "SP-101,Hydraulic Valve,14,125.50\n"
        "SP-202,Bearing Kit,0,42.00\n",
        encoding="utf-8",
    )
    hit = csv_lookup([csv_file], query="SP-101", column="part_id", exact=True)
    assert "Hydraulic Valve" in hit
    assert "row 1" in hit

    # Date & Time
    assert date_tool("weekday", date_a="2026-11-26") == "Thursday"
    assert date_tool("add_days", date_a="2026-11-26", days=5) == "2026-12-01"
    assert date_tool("diff_days", date_a="2026-11-01", date_b="2026-11-11") == "10"
    assert date_tool("business_days", date_a="2026-11-23", date_b="2026-11-27") == "5"

    # HTTP API URL scheme safety
    assert "Invalid or unsupported URL" in http_api("file:///etc/passwd")

    # Web Search missing key behavior
    with patch.dict("os.environ", {}, clear=True):
        assert "not configured" in web_search("mozilla ai")


def test_dynamic_selector_hr_without_csv_selects_retriever_calculator_date(tmp_path: Path):
    docs = tmp_path / "hr_docs"
    docs.mkdir()
    (docs / "leave_policy.txt").write_text(
        "Employees receive 20 days annual leave and 5 business days notice is required.",
        encoding="utf-8",
    )
    req_path = tmp_path / "req.json"
    IntakeInterviewer().run(
        brief="Build an HR assistant for employees about leave policies and holidays.",
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")
    _, topic_cfg = GroundabilityGuard().evaluate(
        requirements=req_path,
        manifest=kb_dir / "knowledge_manifest.json",
        use_llm=False,
    )

    selector = DynamicToolSelector()
    record = selector.select(
        requirements=req_path,
        topic_config=topic_cfg,
        manifest=manifest,
        manifest_path=kb_dir / "knowledge_manifest.json",
        use_llm=False,
        output_dir=tmp_path / "out",
    )

    assert record.selected_tools == ["document_retriever", "calculator", "date_tool"]
    assert "csv_lookup" in record.rejected_tools
    assert "web_search" in record.rejected_tools
    assert all(d.reason for d in record.decisions)
    assert (tmp_path / "out" / "tool_registry.json").is_file()
    assert (tmp_path / "out" / "tool_selection.json").is_file()

    # Bind live document_retriever from registry
    retriever_fn = selector.registry.build_callable(
        "document_retriever",
        manifest_path=kb_dir / "knowledge_manifest.json",
        top_k=2,
    )
    assert "20 days annual leave" in retriever_fn("annual leave")


def test_dynamic_selector_procurement_with_csv_selects_csv_lookup(tmp_path: Path):
    docs = tmp_path / "proc_docs"
    docs.mkdir()
    (docs / "spare_parts_inventory.csv").write_text(
        "part_id,part_name,unit_price,quantity\n"
        "P-100,Gasket,12.50,40\n",
        encoding="utf-8",
    )
    (docs / "procurement_guide.txt").write_text(
        "Procurement Policy:\nCompare vendor unit prices and calculate order totals before approval.",
        encoding="utf-8",
    )

    req_path = tmp_path / "req.json"
    IntakeInterviewer().run(
        brief="Build a procurement assistant to look up spare parts inventory and calculate order costs.",
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")

    selector = DynamicToolSelector()
    record = selector.select(
        requirements=req_path,
        manifest=manifest,
        manifest_path=kb_dir / "knowledge_manifest.json",
        use_llm=False,
    )
    assert set(record.selected_tools) == {"document_retriever", "calculator", "csv_lookup"}
    csv_decision = next(d for d in record.decisions if d.tool_id == "csv_lookup")
    assert csv_decision.selected is True
    assert "CSV" in csv_decision.reason
    assert "Selected: Yes" in csv_decision.format_line()


def test_llm_selection_blocks_unsafe_external_tools_under_closed_book(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Company policy text.", encoding="utf-8")
    req_path = tmp_path / "req.json"
    IntakeInterviewer().run(
        brief="Build an HR policy assistant.",
        docs_dir=docs,
        config={"use_llm": False},
        output_path=req_path,
    )
    kb_dir = tmp_path / "kb"
    manifest = KnowledgeProcessor().process(docs_dir=docs, output_dir=kb_dir, embedding_provider="local")

    # Simulate an LLM trying to turn on web_search despite closed-book restrictions
    mock_batch = LLMToolSelectionBatch(
        decisions=[
            ToolDecision(
                tool_id="web_search",
                display_name="Web Search",
                priority="optional",
                selected=True,
                reason="Want extra info",
            )
        ]
    )
    with patch("forgeai.tools.selector.complete_structured", return_value=mock_batch):
        record = DynamicToolSelector().select(
            requirements=req_path,
            manifest=manifest,
            manifest_path=kb_dir / "knowledge_manifest.json",
            use_llm=True,
        )
        web_dec = next(d for d in record.decisions if d.tool_id == "web_search")
        assert web_dec.selected is False
        assert "closed-book" in web_dec.reason.lower()


def test_tools_cli_list_and_select(tmp_path: Path):
    assert tools_cli_main(["list", "--export", str(tmp_path / "reg.json")]) == 0
    assert (tmp_path / "reg.json").is_file()

