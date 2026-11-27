"""Document chunking strategies and rich metadata attachment."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from forgeai.knowledge.schemas import ChunkMetadata, ChunkingConfig, ExtractedSegment


class DocumentChunker:
    """Split cleaned ExtractedSegments into overlapping chunks with citation metadata."""

    def chunk_segments(
        self,
        segments: list[ExtractedSegment],
        config: ChunkingConfig | None = None,
        id_prefix_override: str | None = None,
    ) -> list[ChunkMetadata]:
        cfg = config or ChunkingConfig()
        if not segments:
            return []

        raw_chunks: list[tuple[ExtractedSegment, str, dict]] = []
        for seg in segments:
            if seg.document_type == "csv" and seg.metadata.get("formatted_rows"):
                row_groups = self._chunk_csv_rows(seg, cfg)
                for text_part, extra in row_groups:
                    raw_chunks.append((seg, text_part, extra))
            elif cfg.strategy == "fixed":
                for text_part in self._chunk_fixed(seg.text, cfg.chunk_size, cfg.chunk_overlap):
                    raw_chunks.append((seg, text_part, {}))
            elif cfg.strategy == "section":
                if len(seg.text) <= cfg.chunk_size:
                    raw_chunks.append((seg, seg.text, {}))
                else:
                    for text_part in self._chunk_recursive(seg.text, cfg.chunk_size, cfg.chunk_overlap):
                        raw_chunks.append((seg, text_part, {}))
            else:
                for text_part in self._chunk_recursive(seg.text, cfg.chunk_size, cfg.chunk_overlap):
                    raw_chunks.append((seg, text_part, {}))

        results: list[ChunkMetadata] = []
        # Maintain unique prefix per relative_path so chunk_ids stay collision-free (e.g. leave_policy_001)
        prefix_counters: dict[str, int] = {}

        for seg, content, extra in raw_chunks:
            cleaned = content.strip()
            if not cleaned:
                continue

            prefix = id_prefix_override or self._slugify(Path(seg.document_name).stem)
            idx = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = idx
            chunk_id = f"{prefix}_{idx:03d}"

            results.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    document_name=seg.document_name,
                    page_number=seg.page_number,
                    section=seg.section,
                    content=cleaned,
                    document_type=seg.document_type,
                    relative_path=seg.relative_path,
                    chunk_index=idx - 1,
                    char_count=len(cleaned),
                    word_count=len(re.findall(r"\b[\w'-]+\b", cleaned)),
                    content_hash=hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16],
                    extra=extra,
                )
            )
        return results

    @staticmethod
    def _chunk_csv_rows(seg: ExtractedSegment, cfg: ChunkingConfig) -> list[tuple[str, dict]]:
        headers: list[str] = seg.metadata.get("headers", [])
        rows: list[str] = seg.metadata.get("formatted_rows", [])
        header_banner = f"Columns: {', '.join(headers)}\n" if headers else ""
        batch_size = cfg.csv_rows_per_chunk

        chunks: list[tuple[str, dict]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            end = start + len(batch)
            content = header_banner + "\n".join(batch)
            # If a batch still exceeds chunk_size significantly, split smaller
            if len(content) > cfg.chunk_size and len(batch) > 1:
                mid = max(1, len(batch) // 2)
                for sub_start in range(0, len(batch), mid):
                    sub = batch[sub_start : sub_start + mid]
                    abs_start = start + sub_start + 1
                    abs_end = abs_start + len(sub) - 1
                    chunks.append(
                        (
                            header_banner + "\n".join(sub),
                            {"row_range": f"{abs_start}-{abs_end}"},
                        )
                    )
            else:
                chunks.append((content, {"row_range": f"{start + 1}-{end}"}))
        return chunks

    @staticmethod
    def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        step = max(1, chunk_size - min(overlap, chunk_size - 1))
        chunks: list[str] = []
        for start in range(0, len(text), step):
            part = text[start : start + chunk_size].strip()
            if part:
                chunks.append(part)
            if start + chunk_size >= len(text):
                break
        return chunks

    @classmethod
    def _chunk_recursive(cls, text: str, chunk_size: int, overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " "]
        units = cls._split_with_best_separator(text, separators, chunk_size)

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current and "\n" in unit else (
                f"{current} {unit}".strip() if current else unit
            )
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(unit) > chunk_size:
                    chunks.extend(cls._chunk_fixed(unit, chunk_size, overlap))
                    current = ""
                else:
                    tail = current[-overlap:].strip() if (overlap > 0 and current) else ""
                    if tail and len(tail) + 1 + len(unit) <= chunk_size:
                        current = f"{tail} {unit}".strip()
                    else:
                        current = unit
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_with_best_separator(text: str, separators: list[str], chunk_size: int) -> list[str]:
        for sep in separators:
            if sep in text:
                parts = [p.strip() for p in text.split(sep) if p.strip()]
                if len(parts) > 1:
                    if sep == ". ":
                        parts = [p + ("." if not p.endswith((".", "!", "?")) else "") for p in parts]
                    return parts
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or "doc"

