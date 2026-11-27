"""Feature 2 — Document Ingestion & Knowledge Processing."""

from forgeai.knowledge.chunker import DocumentChunker
from forgeai.knowledge.cleaner import TextCleaner
from forgeai.knowledge.embeddings import (
    Embedder,
    LiteLLMEmbedder,
    LocalSemanticEmbedder,
    create_embedder,
)
from forgeai.knowledge.extractors import DocumentExtractor
from forgeai.knowledge.loader import SUPPORTED_DOC_EXTENSIONS, DocumentLoader
from forgeai.knowledge.pipeline import KnowledgeProcessor
from forgeai.knowledge.retriever import KnowledgeRetriever, make_retrieval_tool
from forgeai.knowledge.schemas import (
    ChunkMetadata,
    ChunkRecord,
    ChunkingConfig,
    DocumentChunkSummary,
    ExtractedSegment,
    IngestedDocumentInfo,
    KnowledgeManifest,
    LoadedFile,
    RetrievalQuery,
    RetrievalResult,
)
from forgeai.knowledge.vector_store import SQLiteVectorStore

__all__ = [
    "SUPPORTED_DOC_EXTENSIONS",
    "ChunkMetadata",
    "ChunkRecord",
    "ChunkingConfig",
    "DocumentChunkSummary",
    "DocumentChunker",
    "DocumentExtractor",
    "DocumentLoader",
    "Embedder",
    "ExtractedSegment",
    "IngestedDocumentInfo",
    "KnowledgeManifest",
    "KnowledgeProcessor",
    "KnowledgeRetriever",
    "LiteLLMEmbedder",
    "LoadedFile",
    "LocalSemanticEmbedder",
    "RetrievalQuery",
    "RetrievalResult",
    "SQLiteVectorStore",
    "TextCleaner",
    "create_embedder",
    "make_retrieval_tool",
]

