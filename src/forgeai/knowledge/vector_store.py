"""Persistent SQLite + NumPy vector database for chunk embeddings and metadata."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from forgeai.knowledge.schemas import ChunkMetadata, ChunkRecord, DocumentType, RetrievalResult


class SQLiteVectorStore:
    """Lightweight persistent vector store backed by SQLite and NumPy cosine similarity."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    relative_path TEXT,
                    document_type TEXT NOT NULL,
                    page_number INTEGER,
                    section TEXT,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    extra_json TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(document_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks(document_name)")

    def set_metadata(self, meta: dict[str, Any]) -> None:
        with self._connect() as conn:
            for key, value in meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                    (str(key), json.dumps(value) if not isinstance(value, str) else value),
                )

    def get_metadata(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM store_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")

    def delete_by_document(self, document_name: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE document_name = ?", (document_name,))
            return cur.rowcount

    def upsert_chunks(self, records: list[ChunkRecord]) -> int:
        if not records:
            return 0
        with self._connect() as conn:
            for record in records:
                meta = record.metadata
                emb_bytes = np.asarray(record.embedding, dtype=np.float32).tobytes()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, document_name, relative_path, document_type,
                        page_number, section, content, chunk_index,
                        char_count, word_count, content_hash, extra_json, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.chunk_id,
                        meta.document_name,
                        meta.relative_path,
                        meta.document_type,
                        meta.page_number,
                        meta.section,
                        meta.content,
                        meta.chunk_index,
                        meta.char_count,
                        meta.word_count,
                        meta.content_hash,
                        json.dumps(meta.extra),
                        emb_bytes,
                    ),
                )
        return len(records)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        return int(row["cnt"]) if row else 0

    def list_chunks(self) -> list[ChunkMetadata]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY document_name, chunk_index").fetchall()
        return [self._row_to_metadata(row) for row in rows]

    def query(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 5,
        document_type: DocumentType | None = None,
        document_name: str | None = None,
        section: str | None = None,
        page_number: int | None = None,
        min_score: float = 0.0,
        hybrid_keyword_weight: float = 0.2,
    ) -> list[RetrievalResult]:
        clauses: list[str] = []
        params: list[Any] = []
        if document_type is not None:
            clauses.append("document_type = ?")
            params.append(document_type)
        if document_name is not None:
            clauses.append("LOWER(document_name) = LOWER(?)")
            params.append(document_name)
        if section is not None:
            clauses.append("LOWER(section) LIKE LOWER(?)")
            params.append(f"%{section}%")
        if page_number is not None:
            clauses.append("page_number = ?")
            params.append(page_number)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM chunks {where_sql}", params).fetchall()

        if not rows:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec /= q_norm

        matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        matrix_normed = matrix / norms
        semantic_scores = matrix_normed @ q_vec

        query_terms = set(re.findall(r"[a-z0-9]{2,}", query_text.lower()))
        results: list[RetrievalResult] = []
        for idx, row in enumerate(rows):
            sem_score = float(semantic_scores[idx])
            kw_score = self._keyword_overlap_score(query_terms, row["content"], row["section"] or "")
            combined = (1.0 - hybrid_keyword_weight) * sem_score + hybrid_keyword_weight * kw_score
            if combined < min_score:
                continue
            chunk_meta = self._row_to_metadata(row)
            results.append(
                RetrievalResult(
                    chunk=chunk_meta,
                    score=round(combined, 4),
                    semantic_score=round(sem_score, 4),
                    keyword_score=round(kw_score, 4),
                    citation=self._format_citation(chunk_meta),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _keyword_overlap_score(query_terms: set[str], content: str, section: str) -> float:
        if not query_terms:
            return 0.0
        target_terms = set(re.findall(r"[a-z0-9]{2,}", f"{section} {content}".lower()))
        if not target_terms:
            return 0.0
        matches = len(query_terms & target_terms)
        return matches / len(query_terms)

    @staticmethod
    def _format_citation(meta: ChunkMetadata) -> str:
        parts = [meta.document_name]
        if meta.page_number is not None:
            parts.append(f"Page {meta.page_number}")
        if meta.section:
            parts.append(f"Section: {meta.section}")
        return " | ".join(parts)

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id=row["chunk_id"],
            document_name=row["document_name"],
            page_number=row["page_number"],
            section=row["section"],
            content=row["content"],
            document_type=row["document_type"],
            relative_path=row["relative_path"],
            chunk_index=row["chunk_index"],
            char_count=row["char_count"],
            word_count=row["word_count"],
            content_hash=row["content_hash"],
            extra=json.loads(row["extra_json"]) if row["extra_json"] else {},
        )

