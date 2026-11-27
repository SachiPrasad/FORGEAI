"""Comprehensive tests for Feature 2 — Document Ingestion & Knowledge Processing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from docx import Document

from forgeai.knowledge import (
    ChunkingConfig,
    DocumentChunker,
    DocumentExtractor,
    DocumentLoader,
    KnowledgeManifest,
    KnowledgeProcessor,
    KnowledgeRetriever,
    LiteLLMEmbedder,
    LocalSemanticEmbedder,
    SQLiteVectorStore,
    TextCleaner,
    make_retrieval_tool,
)
from forgeai.knowledge.cli import main as knowledge_cli_main


def _write_multipage_pdf(path: Path, pages: list[list[str]]) -> None:
    """Create a valid multi-page PDF with extractable Helvetica text per page."""
    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font
    page_obj_nums = [4 + i * 2 for i in range(len(pages))]
    kids_str = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >>\nendobj\n".encode("latin-1"))
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    for i, lines in enumerate(pages):
        page_num = 4 + i * 2
        content_num = page_num + 1
        escaped = [l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for l in lines]
        cmds = ["BT", "/F1 12 Tf", "72 720 Td"]
        for idx, l in enumerate(escaped):
            if idx > 0:
                cmds.append("0 -18 Td")
            cmds.append(f"({l}) Tj")
        cmds.append("ET")
        stream = "\n".join(cmds).encode("latin-1")
        objects.append(
            f"{page_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>\nendobj\n".encode("latin-1")
        )
        objects.append(
            f"{content_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream\nendobj\n"
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(bytes(pdf))


@pytest.fixture()
def sample_kb_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()

    _write_multipage_pdf(
        docs / "leave_policy.pdf",
        [
            ["Leave Overview", "All full-time employees are covered by company leave rules."],
            ["Annual Leave", "Employees are eligible for 20 days of paid annual leave per year."],
        ],
    )

    docx_doc = Document()
    docx_doc.add_heading("Health Insurance", level=1)
    docx_doc.add_paragraph("Medical, dental, and vision coverage starts on the first day of the next month.")
    docx_doc.add_heading("Wellness Stipend", level=1)
    docx_doc.add_paragraph("Employees receive a 500 USD annual wellness stipend.")
    docx_doc.save(str(docs / "benefits_guide.docx"))

    (docs / "hr_handbook.txt").write_text(
        "Workplace Conduct:\nEmployees must protect confidential data and maintain respectful behavior.",
        encoding="utf-8",
    )

    (docs / "attendance_rules.csv").write_text(
        "rule_id,category,rule_summary\n"
        "ATT-01,Core Hours,Core collaboration window is 10am to 4pm\n"
        "ATT-02,Remote Work,Up to 2 remote work days per week\n",
        encoding="utf-8",
    )

    (docs / "company_faq.md").write_text(
        "# FAQ\n\n## Public Holidays\nThe company observes 12 paid public holidays annually.\n",
        encoding="utf-8",
    )

    (docs / "unsupported.bin").write_bytes(b"\x00\x01\x02")
    return docs


def test_loader_extracts_all_five_formats_and_skips_unsupported(sample_kb_docs: Path):
    loaded, skipped = DocumentLoader().load_folder(sample_kb_docs)
    assert len(loaded) == 5
    assert skipped == ["unsupported.bin"]
    assert {f.document_type for f in loaded} == {"pdf", "docx", "txt", "csv", "md"}


def test_extractor_preserves_page_numbers_and_sections(sample_kb_docs: Path):
    loaded, _ = DocumentLoader().load_folder(sample_kb_docs)
    by_type = {f.document_type: f for f in loaded}
    extractor = DocumentExtractor()

    pdf_info = extractor.extract(by_type["pdf"])
    assert pdf_info.page_count == 2
    assert [seg.page_number for seg in pdf_info.segments] == [1, 2]
    assert pdf_info.segments[1].section == "Annual Leave"

    docx_info = extractor.extract(by_type["docx"])
    assert [seg.section for seg in docx_info.segments] == ["Health Insurance", "Wellness Stipend"]

    md_info = extractor.extract(by_type["md"])
    assert md_info.segments[0].section == "Public Holidays"

    csv_info = extractor.extract(by_type["csv"])
    assert csv_info.row_count == 2
    assert "ATT-02" in csv_info.segments[0].text


def test_text_cleaner_fixes_hyphenation_and_whitespace():
    cleaner = TextCleaner()
    dirty = "Employ-\nees\u00a0receive   annual\t\tleave.\r\n\r\n\r\nNext   line."
    cleaned = cleaner.clean_text(dirty)
    assert cleaned == "Employees receive annual leave.\n\nNext line."


def test_chunker_metadata_and_strategies(sample_kb_docs: Path):
    loaded, _ = DocumentLoader().load_folder(sample_kb_docs)
    by_name = {f.document_name: f for f in loaded}
    extractor = DocumentExtractor()
    cleaner = TextCleaner()
    chunker = DocumentChunker()

    pdf_segs = cleaner.clean_segments(extractor.extract(by_name["leave_policy.pdf"]).segments)
    chunks = chunker.chunk_segments(pdf_segs, ChunkingConfig(strategy="recursive", chunk_size=250, chunk_overlap=30))
    assert len(chunks) == 2
    second = chunks[1]
    assert second.chunk_id == "leave_policy_002"
    assert second.document_name == "leave_policy.pdf"
    assert second.page_number == 2
    assert second.section == "Annual Leave"
    assert second.document_type == "pdf"
    assert "20 days of paid annual leave" in second.content

    # Fixed chunking strategy
    fixed = chunker.chunk_segments(pdf_segs, ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=20))
    assert len(fixed) >= 2


def test_pipeline_creates_manifest_and_retrieves_with_citations(sample_kb_docs: Path, tmp_path: Path):
    kb_dir = tmp_path / "kb"
    processor = KnowledgeProcessor()
    manifest = processor.process(
        docs_dir=sample_kb_docs,
        output_dir=kb_dir,
        embedding_provider="local",
    )

    manifest_path = kb_dir / "knowledge_manifest.json"
    db_path = kb_dir / "knowledge_base.sqlite"
    assert manifest_path.is_file()
    assert db_path.is_file()
    assert manifest.total_documents == 5
    assert manifest.total_chunks >= 6

    # Validate manifest schema roundtrip
    parsed_manifest = KnowledgeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert parsed_manifest.total_chunks == manifest.total_chunks
    assert parsed_manifest.chunks[0].model_dump().keys() >= {
        "chunk_id",
        "document_name",
        "page_number",
        "section",
        "content",
        "document_type",
    }

    # Semantic retrieval via manifest
    retriever = KnowledgeRetriever.from_manifest(manifest_path)
    leave_hits = retriever.retrieve("How many days of paid annual leave do employees get?", top_k=3)
    assert leave_hits
    assert leave_hits[0].chunk.document_name == "leave_policy.pdf"
    assert leave_hits[0].chunk.page_number == 2
    assert leave_hits[0].chunk.section == "Annual Leave"
    assert "Page 2" in leave_hits[0].citation

    # Metadata filter check
    csv_hits = retriever.retrieve("remote work days", top_k=2, document_type="csv")
    assert len(csv_hits) == 1
    assert csv_hits[0].chunk.document_type == "csv"

    # Agent-ready retrieval tool wrapper
    tool_fn = make_retrieval_tool(manifest_path)
    tool_output = tool_fn("wellness stipend USD", 2)
    assert "benefits_guide.docx" in tool_output
    assert "500 USD" in tool_output


def test_litellm_embedder_mocked():
    fake_response = SimpleNamespace(
        data=[
            {"embedding": [3.0, 4.0, 0.0]},
            {"embedding": [0.0, 5.0, 12.0]},
        ]
    )
    with patch("litellm.embedding", return_value=fake_response) as mock_emb:
        embedder = LiteLLMEmbedder(model_name="text-embedding-3-small")
        vecs = embedder.embed_documents(["hello", "world"])
        mock_emb.assert_called_once()
        assert embedder.dimension == 3
        assert vecs[0] == pytest.approx([0.6, 0.8, 0.0], abs=1e-5)


def test_knowledge_cli_ingest_and_query(sample_kb_docs: Path, tmp_path: Path):
    kb_dir = tmp_path / "cli_kb"
    assert (
        knowledge_cli_main(
            [
                "ingest",
                "--docs",
                str(sample_kb_docs),
                "--out",
                str(kb_dir),
                "--provider",
                "local",
            ]
        )
        == 0
    )
    manifest_file = kb_dir / "knowledge_manifest.json"
    assert manifest_file.is_file()

    assert (
        knowledge_cli_main(
            [
                "query",
                "--manifest",
                str(manifest_file),
                "--q",
                "public holidays",
                "--top-k",
                "2",
            ]
        )
        == 0
    )

