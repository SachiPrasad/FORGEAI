"""Pydantic v2 schemas for Feature 2 — Document Ingestion & Knowledge Processing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentType = Literal["pdf", "docx", "txt", "csv", "md"]
ChunkingStrategy = Literal["recursive", "section", "fixed", "csv_rows"]
EmbeddingProvider = Literal["auto", "local", "litellm"]


class LoadedFile(BaseModel):
    """Represents a discovered supported file ready for extraction."""

    document_name: str
    relative_path: str
    absolute_path: str
    document_type: DocumentType
    size_bytes: int
    sha256: str


class ExtractedSegment(BaseModel):
    """Atomic extracted unit preserving structural metadata (page, section, row range)."""

    document_name: str
    relative_path: str
    document_type: DocumentType
    text: str
    page_number: int | None = None
    section: str | None = None
    row_range: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    strategy: ChunkingStrategy = "recursive"
    chunk_size: int = Field(default=600, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    csv_rows_per_chunk: int = Field(default=10, ge=1, le=200)


class ChunkMetadata(BaseModel):
    """Structured chunk payload with rich citation metadata (recommended schema)."""

    chunk_id: str
    document_name: str
    page_number: int | None = None
    section: str | None = None
    content: str
    document_type: DocumentType
    relative_path: str | None = None
    chunk_index: int = 0
    char_count: int = 0
    word_count: int = 0
    content_hash: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """Chunk paired with its dense embedding vector."""

    metadata: ChunkMetadata
    embedding: list[float]


class DocumentChunkSummary(BaseModel):
    """Per-document ingestion summary recorded in knowledge_manifest.json."""

    document_name: str
    relative_path: str
    document_type: DocumentType
    sha256: str
    size_bytes: int
    page_count: int | None = None
    row_count: int | None = None
    sections_detected: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IngestedDocumentInfo(BaseModel):
    """Extraction result for one file before chunking."""

    file: LoadedFile
    segments: list[ExtractedSegment] = Field(default_factory=list)
    page_count: int | None = None
    row_count: int | None = None
    warnings: list[str] = Field(default_factory=list)


class KnowledgeManifest(BaseModel):
    """Top-level manifest emitted alongside the searchable vector database."""

    manifest_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    docs_dir: str
    vector_store_backend: str = "sqlite_numpy"
    vector_store_path: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chunking: ChunkingConfig
    total_documents: int
    total_chunks: int
    skipped_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    kb_validation: dict[str, Any] = Field(default_factory=dict)
    documents: list[DocumentChunkSummary] = Field(default_factory=list)
    chunks: list[ChunkMetadata] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    """Query specification for semantic retrieval."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_type: DocumentType | None = None
    document_name: str | None = None
    section: str | None = None
    page_number: int | None = None
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    hybrid_keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    """Single semantic search hit with citation metadata and similarity score."""

    chunk: ChunkMetadata
    score: float
    semantic_score: float
    keyword_score: float = 0.0
    citation: str

