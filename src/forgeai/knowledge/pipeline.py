"""End-to-end Feature 2 pipeline: Load -> Extract -> Clean -> Chunk -> Embed -> Store -> Manifest."""

from __future__ import annotations

from pathlib import Path

from forgeai.knowledge.chunker import DocumentChunker
from forgeai.knowledge.cleaner import TextCleaner
from forgeai.knowledge.embeddings import Embedder, create_embedder
from forgeai.knowledge.extractors import DocumentExtractor
from forgeai.knowledge.loader import DocumentLoader
from forgeai.knowledge.schemas import (
    ChunkMetadata,
    ChunkRecord,
    ChunkingConfig,
    DocumentChunkSummary,
    EmbeddingProvider,
    IngestedDocumentInfo,
    KnowledgeManifest,
)
from forgeai.knowledge.validator import KnowledgeBaseValidator
from forgeai.knowledge.vector_store import SQLiteVectorStore


class KnowledgeProcessor:
    """Orchestrates document ingestion and outputs knowledge_manifest.json + searchable vector DB."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        extractor: DocumentExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.extractor = extractor or DocumentExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or DocumentChunker()

    def process(
        self,
        docs_dir: str | Path,
        output_dir: str | Path,
        chunking: ChunkingConfig | None = None,
        embedding_provider: EmbeddingProvider = "auto",
        embedding_model: str | None = None,
        embedding_dimension: int = 384,
        include_subdirectories: bool = True,
        reset_store: bool = True,
        db_filename: str = "knowledge_base.sqlite",
        manifest_filename: str = "knowledge_manifest.json",
    ) -> KnowledgeManifest:
        cfg = chunking or ChunkingConfig()
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        db_path = out_root / db_filename
        manifest_path = out_root / manifest_filename

        loaded_files, skipped_files = self.loader.load_folder(
            docs_dir,
            include_subdirectories=include_subdirectories,
        )

        all_chunks: list[ChunkMetadata] = []
        doc_summaries: list[DocumentChunkSummary] = []
        extracted_infos: list[IngestedDocumentInfo] = []
        warnings: list[str] = []

        # Maintain unique chunk_id numbering across all documents even if stems collide
        global_stem_counts: dict[str, int] = {}

        for loaded in loaded_files:
            extracted = self.extractor.extract(loaded)
            extracted_infos.append(extracted)
            warnings.extend(extracted.warnings)

            cleaned_segments = self.cleaner.clean_segments(extracted.segments)
            stem_slug = self.chunker._slugify(Path(loaded.document_name).stem)
            prefix_override = None
            if stem_slug in global_stem_counts:
                global_stem_counts[stem_slug] += 1
                prefix_override = f"{stem_slug}_{global_stem_counts[stem_slug]}"
            else:
                global_stem_counts[stem_slug] = 1

            chunks = self.chunker.chunk_segments(
                cleaned_segments,
                config=cfg,
                id_prefix_override=prefix_override,
            )
            all_chunks.extend(chunks)

            sections = self._dedupe([c.section for c in chunks if c.section])
            doc_summaries.append(
                DocumentChunkSummary(
                    document_name=loaded.document_name,
                    relative_path=loaded.relative_path,
                    document_type=loaded.document_type,
                    sha256=loaded.sha256,
                    size_bytes=loaded.size_bytes,
                    page_count=extracted.page_count,
                    row_count=extracted.row_count,
                    sections_detected=sections,
                    chunk_count=len(chunks),
                    chunk_ids=[c.chunk_id for c in chunks],
                    warnings=extracted.warnings,
                )
            )

        if not loaded_files:
            warnings.append(f"No supported documents found in {docs_dir}.")

        embedder: Embedder = create_embedder(
            provider=embedding_provider,
            model=embedding_model,
            dimension=embedding_dimension,
        )

        texts = [c.content for c in all_chunks]
        vectors = embedder.embed_documents(texts) if texts else []
        actual_dim = len(vectors[0]) if vectors else embedder.dimension

        store = SQLiteVectorStore(db_path)
        if reset_store:
            store.clear()
        store.set_metadata(
            {
                "embedding_provider": embedder.provider_name,
                "embedding_model": embedder.model_name,
                "embedding_dimension": str(actual_dim),
            }
        )
        records = [ChunkRecord(metadata=chunk, embedding=vec) for chunk, vec in zip(all_chunks, vectors, strict=True)]
        store.upsert_chunks(records)

        kb_validation = KnowledgeBaseValidator().validate(loaded_files, extracted_infos)
        for issue in kb_validation.issues:
            if issue.message not in warnings:
                warnings.append(issue.message)

        manifest = KnowledgeManifest(
            docs_dir=str(Path(docs_dir).resolve()),
            vector_store_backend="sqlite_numpy",
            vector_store_path=db_filename,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
            embedding_dimension=actual_dim,
            chunking=cfg,
            total_documents=len(doc_summaries),
            total_chunks=len(all_chunks),
            skipped_files=skipped_files,
            warnings=warnings,
            kb_validation=kb_validation.model_dump(),
            documents=doc_summaries,
            chunks=all_chunks,
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

