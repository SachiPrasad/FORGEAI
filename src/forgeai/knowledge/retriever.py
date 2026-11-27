"""Semantic retrieval and agent-callable knowledge tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from forgeai.knowledge.embeddings import Embedder, create_embedder
from forgeai.knowledge.schemas import DocumentType, KnowledgeManifest, RetrievalQuery, RetrievalResult
from forgeai.knowledge.vector_store import SQLiteVectorStore


class KnowledgeRetriever:
    """High-level semantic + metadata-filtered retriever over a processed knowledge base."""

    def __init__(self, vector_store: SQLiteVectorStore, embedder: Embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> KnowledgeRetriever:
        manifest_file = Path(manifest_path)
        manifest = KnowledgeManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        db_path = Path(manifest.vector_store_path)
        if not db_path.is_absolute():
            db_path = manifest_file.parent / db_path

        provider = "local" if manifest.embedding_provider == "local" else "litellm"
        embedder = create_embedder(
            provider=provider,
            model=manifest.embedding_model,
            dimension=manifest.embedding_dimension,
        )
        return cls(vector_store=SQLiteVectorStore(db_path), embedder=embedder)

    def retrieve(
        self,
        query: str | RetrievalQuery,
        top_k: int = 5,
        document_type: DocumentType | None = None,
        document_name: str | None = None,
        section: str | None = None,
        page_number: int | None = None,
        min_score: float = 0.0,
        hybrid_keyword_weight: float = 0.2,
    ) -> list[RetrievalResult]:
        if isinstance(query, RetrievalQuery):
            req = query
        else:
            req = RetrievalQuery(
                query=query,
                top_k=top_k,
                document_type=document_type,
                document_name=document_name,
                section=section,
                page_number=page_number,
                min_score=min_score,
                hybrid_keyword_weight=hybrid_keyword_weight,
            )

        q_embedding = self.embedder.embed_query(req.query)
        return self.vector_store.query(
            query_embedding=q_embedding,
            query_text=req.query,
            top_k=req.top_k,
            document_type=req.document_type,
            document_name=req.document_name,
            section=req.section,
            page_number=req.page_number,
            min_score=req.min_score,
            hybrid_keyword_weight=req.hybrid_keyword_weight,
        )

    @staticmethod
    def format_results(results: list[RetrievalResult]) -> str:
        """Format retrieved chunks with citations for LLM/agent consumption."""
        if not results:
            return "No relevant knowledge base entries found."
        blocks: list[str] = []
        for idx, res in enumerate(results, start=1):
            blocks.append(
                f"[{idx}] ({res.citation} | chunk_id={res.chunk.chunk_id} | score={res.score:.3f})\n"
                f"{res.chunk.content}"
            )
        return "\n\n---\n\n".join(blocks)


def make_retrieval_tool(manifest_path: str | Path, default_top_k: int = 5) -> Callable[[str, int], str]:
    """Create a standalone Python tool function compatible with any-agent / agent-factory."""
    retriever = KnowledgeRetriever.from_manifest(manifest_path)

    def retrieve_knowledge(query: str, top_k: int = default_top_k) -> str:
        """Search the ingested company knowledge base and return cited excerpts.

        Args:
            query: Natural language question or search query.
            top_k: Maximum number of relevant document chunks to return.

        Returns:
            Formatted text containing top matching chunks with document, page, and section citations.
        """
        results = retriever.retrieve(query=query, top_k=top_k)
        return retriever.format_results(results)

    return retrieve_knowledge

