"""Format-specific document extractors preserving page numbers and sections."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from forgeai.knowledge.schemas import ExtractedSegment, IngestedDocumentInfo, LoadedFile


class DocumentExtractor:
    """Extract structured segments from PDF, DOCX, TXT, CSV, and Markdown files."""

    def extract(self, loaded_file: LoadedFile) -> IngestedDocumentInfo:
        path = Path(loaded_file.absolute_path)
        doc_type = loaded_file.document_type

        if doc_type == "pdf":
            return self._extract_pdf(loaded_file, path)
        if doc_type == "docx":
            return self._extract_docx(loaded_file, path)
        if doc_type == "csv":
            return self._extract_csv(loaded_file, path)
        if doc_type == "md":
            return self._extract_markdown(loaded_file, path)
        return self._extract_txt(loaded_file, path)

    def _extract_pdf(self, loaded: LoadedFile, path: Path) -> IngestedDocumentInfo:
        warnings: list[str] = []
        segments: list[ExtractedSegment] = []
        page_count: int | None = None

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = (page.extract_text() or "").strip()
                if not raw_text:
                    continue
                section = self._guess_page_section(raw_text) or f"Page {page_idx}"
                segments.append(
                    ExtractedSegment(
                        document_name=loaded.document_name,
                        relative_path=loaded.relative_path,
                        document_type="pdf",
                        text=raw_text,
                        page_number=page_idx,
                        section=section,
                    )
                )
            if not segments:
                warnings.append(f"PDF '{loaded.relative_path}' yielded no extractable text.")
        except ImportError:
            warnings.append(f"pypdf is not installed; skipped PDF '{loaded.relative_path}'.")
        except Exception as exc:
            warnings.append(f"Failed to extract PDF '{loaded.relative_path}': {exc}")

        return IngestedDocumentInfo(
            file=loaded,
            segments=segments,
            page_count=page_count,
            warnings=warnings,
        )

    def _extract_docx(self, loaded: LoadedFile, path: Path) -> IngestedDocumentInfo:
        warnings: list[str] = []
        segments: list[ExtractedSegment] = []

        try:
            from docx import Document

            doc = Document(str(path))
            current_section: str | None = None
            buffer: list[str] = []

            def flush_buffer() -> None:
                if buffer:
                    segments.append(
                        ExtractedSegment(
                            document_name=loaded.document_name,
                            relative_path=loaded.relative_path,
                            document_type="docx",
                            text="\n".join(buffer),
                            page_number=None,
                            section=current_section or path.stem.replace("_", " ").title(),
                        )
                    )
                    buffer.clear()

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                style_name = (para.style.name if para.style else "") or ""
                if style_name.lower().startswith("heading"):
                    flush_buffer()
                    current_section = text
                else:
                    buffer.append(text)

            for table_idx, table in enumerate(doc.tables, start=1):
                rows_text: list[str] = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    buffer.append(f"[Table {table_idx}]\n" + "\n".join(rows_text))

            flush_buffer()
        except ImportError:
            # Stdlib XML fallback
            try:
                with zipfile.ZipFile(path) as archive:
                    xml_bytes = archive.read("word/document.xml")
                root = ET.fromstring(xml_bytes)
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                lines: list[str] = []
                for para in root.iter(f"{ns}p"):
                    line = "".join(n.text for n in para.iter(f"{ns}t") if n.text).strip()
                    if line:
                        lines.append(line)
                if lines:
                    segments.append(
                        ExtractedSegment(
                            document_name=loaded.document_name,
                            relative_path=loaded.relative_path,
                            document_type="docx",
                            text="\n".join(lines),
                            section=path.stem.replace("_", " ").title(),
                        )
                    )
            except Exception as exc:
                warnings.append(f"Failed to extract DOCX '{loaded.relative_path}': {exc}")
        except Exception as exc:
            warnings.append(f"Failed to extract DOCX '{loaded.relative_path}': {exc}")

        if not segments and not warnings:
            warnings.append(f"DOCX '{loaded.relative_path}' yielded no text.")

        return IngestedDocumentInfo(file=loaded, segments=segments, warnings=warnings)

    def _extract_csv(self, loaded: LoadedFile, path: Path) -> IngestedDocumentInfo:
        warnings: list[str] = []
        segments: list[ExtractedSegment] = []
        rows: list[list[str]] = []

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    rows = list(csv.reader(handle))
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                warnings.append(f"Failed to read CSV '{loaded.relative_path}': {exc}")
                return IngestedDocumentInfo(file=loaded, warnings=warnings)

        if not rows:
            warnings.append(f"CSV '{loaded.relative_path}' is empty.")
            return IngestedDocumentInfo(file=loaded, row_count=0, warnings=warnings)

        headers = [h.strip() or f"col_{i}" for i, h in enumerate(rows[0], start=1)]
        data_rows = rows[1:]
        if not data_rows:
            segments.append(
                ExtractedSegment(
                    document_name=loaded.document_name,
                    relative_path=loaded.relative_path,
                    document_type="csv",
                    text=f"Columns: {', '.join(headers)} (0 data rows)",
                    section="CSV Schema",
                    row_range="0-0",
                    metadata={"headers": headers, "rows": []},
                )
            )
            return IngestedDocumentInfo(file=loaded, segments=segments, row_count=0, warnings=warnings)

        formatted_rows: list[str] = []
        for idx, row in enumerate(data_rows, start=1):
            pairs = [
                f"{headers[col_i]}: {row[col_i].strip()}"
                for col_i in range(min(len(headers), len(row)))
                if row[col_i].strip()
            ]
            if pairs:
                formatted_rows.append(f"Row {idx}: " + " | ".join(pairs))

        # Store single structured CSV segment plus headers/row strings in metadata for csv_rows chunker
        segments.append(
            ExtractedSegment(
                document_name=loaded.document_name,
                relative_path=loaded.relative_path,
                document_type="csv",
                text="\n".join(formatted_rows),
                section=path.stem.replace("_", " ").title(),
                row_range=f"1-{len(data_rows)}",
                metadata={"headers": headers, "formatted_rows": formatted_rows},
            )
        )
        return IngestedDocumentInfo(
            file=loaded,
            segments=segments,
            row_count=len(data_rows),
            warnings=warnings,
        )

    def _extract_markdown(self, loaded: LoadedFile, path: Path) -> IngestedDocumentInfo:
        raw, warn = self._read_text_resilient(path)
        warnings = [warn] if warn else []
        segments: list[ExtractedSegment] = []

        if raw.strip():
            current_section = path.stem.replace("_", " ").title()
            buffer: list[str] = []
            heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

            for line in raw.splitlines():
                match = heading_pattern.match(line.strip())
                if match:
                    if buffer and "\n".join(buffer).strip():
                        segments.append(
                            ExtractedSegment(
                                document_name=loaded.document_name,
                                relative_path=loaded.relative_path,
                                document_type="md",
                                text="\n".join(buffer).strip(),
                                section=current_section,
                            )
                        )
                        buffer.clear()
                    current_section = match.group(2).strip()
                else:
                    buffer.append(line)

            if buffer and "\n".join(buffer).strip():
                segments.append(
                    ExtractedSegment(
                        document_name=loaded.document_name,
                        relative_path=loaded.relative_path,
                        document_type="md",
                        text="\n".join(buffer).strip(),
                        section=current_section,
                    )
                )

        if not segments and not warnings:
            warnings.append(f"Markdown '{loaded.relative_path}' yielded no text.")
        return IngestedDocumentInfo(file=loaded, segments=segments, warnings=warnings)

    def _extract_txt(self, loaded: LoadedFile, path: Path) -> IngestedDocumentInfo:
        raw, warn = self._read_text_resilient(path)
        warnings = [warn] if warn else []
        segments: list[ExtractedSegment] = []

        if raw.strip():
            current_section = path.stem.replace("_", " ").title()
            buffer: list[str] = []

            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    if buffer and "\n".join(buffer).strip():
                        segments.append(
                            ExtractedSegment(
                                document_name=loaded.document_name,
                                relative_path=loaded.relative_path,
                                document_type="txt",
                                text="\n".join(buffer).strip(),
                                section=current_section,
                            )
                        )
                        buffer.clear()
                    current_section = stripped.lstrip("# ").strip()
                elif stripped.endswith(":") and len(stripped.split()) <= 6:
                    if buffer and "\n".join(buffer).strip():
                        segments.append(
                            ExtractedSegment(
                                document_name=loaded.document_name,
                                relative_path=loaded.relative_path,
                                document_type="txt",
                                text="\n".join(buffer).strip(),
                                section=current_section,
                            )
                        )
                        buffer.clear()
                    current_section = stripped.rstrip(":")
                else:
                    buffer.append(line)

            if buffer and "\n".join(buffer).strip():
                segments.append(
                    ExtractedSegment(
                        document_name=loaded.document_name,
                        relative_path=loaded.relative_path,
                        document_type="txt",
                        text="\n".join(buffer).strip(),
                        section=current_section,
                    )
                )

        if not segments and not warnings:
            warnings.append(f"Text file '{loaded.relative_path}' yielded no text.")
        return IngestedDocumentInfo(file=loaded, segments=segments, warnings=warnings)

    @staticmethod
    def _read_text_resilient(path: Path) -> tuple[str, str | None]:
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return path.read_text(encoding=encoding), None
            except UnicodeDecodeError:
                continue
        return "", f"Could not decode text file '{path.name}'."

    @staticmethod
    def _guess_page_section(page_text: str) -> str | None:
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) <= 75 and not stripped.endswith("."):
                return stripped
        return None

