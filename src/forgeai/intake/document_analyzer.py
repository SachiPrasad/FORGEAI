"""Document analyzer supporting PDF, DOCX, TXT, and CSV folders."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Literal

from forgeai.intake.schemas import DocumentAnalysisResult, DocumentSummary

SUPPORTED_EXTENSIONS: dict[str, Literal["pdf", "docx", "txt", "csv"]] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".csv": "csv",
}

_COMMON_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "are",
    "was",
    "were",
    "can",
    "not",
    "all",
    "any",
    "you",
    "your",
    "our",
    "per",
    "must",
    "may",
    "should",
    "days",
    "employees",
    "employee",
    "company",
    "policy",
}


class DocumentAnalyzer:
    """Scan a document folder and extract structured text, metadata, and topic cues."""

    def analyze_folder(
        self,
        folder_path: str | Path,
        include_subdirectories: bool = True,
    ) -> DocumentAnalysisResult:
        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(f"Documents folder does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected a directory for documents, got: {root}")

        pattern = "**/*" if include_subdirectories else "*"
        all_files = sorted(p for p in root.glob(pattern) if p.is_file())

        documents: list[DocumentSummary] = []
        skipped_files: list[str] = []
        warnings: list[str] = []

        for file_path in all_files:
            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped_files.append(rel_path)
                continue

            doc_summary = self.analyze_file(file_path, relative_path=rel_path)
            documents.append(doc_summary)
            warnings.extend(doc_summary.warnings)

        if not documents:
            warnings.append(f"No supported documents (PDF, DOCX, TXT, CSV) found in {root}.")

        available_sources = [doc.source_label for doc in documents]
        combined_topics = self._dedupe_preserve_order(
            [topic for doc in documents for topic in doc.key_topics]
        )

        return DocumentAnalysisResult(
            folder_path=str(root.resolve()),
            total_files_scanned=len(all_files),
            supported_files_count=len(documents),
            documents=documents,
            skipped_files=skipped_files,
            available_sources=available_sources,
            combined_topics=combined_topics[:25],
            warnings=warnings,
        )

    def analyze_file(self, file_path: Path, relative_path: str | None = None) -> DocumentSummary:
        ext = file_path.suffix.lower()
        doc_type = SUPPORTED_EXTENSIONS[ext]
        rel = relative_path or file_path.name
        warnings: list[str] = []
        page_count: int | None = None
        row_count: int | None = None
        columns: list[str] = []
        text = ""

        if doc_type == "txt":
            text, warn = self._read_txt(file_path)
            if warn:
                warnings.append(warn)
        elif doc_type == "csv":
            text, columns, row_count, warn = self._read_csv(file_path)
            if warn:
                warnings.append(warn)
        elif doc_type == "docx":
            text, warn = self._read_docx(file_path)
            if warn:
                warnings.append(warn)
        elif doc_type == "pdf":
            text, page_count, warn = self._read_pdf(file_path)
            if warn:
                warnings.append(warn)

        cleaned_text = re.sub(r"\r\n?", "\n", text).strip()
        if not cleaned_text:
            warnings.append(f"Document '{rel}' yielded empty text.")

        headings = self._extract_headings(cleaned_text, columns=columns)
        source_label = self._infer_source_label(file_path.stem, headings)
        key_topics = self._extract_key_topics(cleaned_text, headings, columns)
        words = re.findall(r"\b[\w'-]+\b", cleaned_text)

        return DocumentSummary(
            filename=file_path.name,
            relative_path=rel,
            doc_type=doc_type,
            source_label=source_label,
            char_count=len(cleaned_text),
            word_count=len(words),
            page_count=page_count,
            row_count=row_count,
            columns=columns,
            detected_headings=headings,
            key_topics=key_topics,
            excerpt=cleaned_text[:500],
            full_text=cleaned_text,
            warnings=warnings,
        )

    @staticmethod
    def _read_txt(path: Path) -> tuple[str, str | None]:
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return path.read_text(encoding=encoding), None
            except UnicodeDecodeError:
                continue
        return "", f"Could not decode text file '{path.name}'."

    @staticmethod
    def _read_csv(path: Path) -> tuple[str, list[str], int, str | None]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
        except UnicodeDecodeError:
            with path.open("r", encoding="latin-1", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
        except Exception as exc:
            return "", [], 0, f"Failed to parse CSV '{path.name}': {exc}"

        if not rows:
            return "", [], 0, f"CSV file '{path.name}' is empty."

        headers = [h.strip() for h in rows[0] if h.strip()]
        data_rows = rows[1:]
        sample_lines = [", ".join(row) for row in rows[:25]]
        summary_text = (
            f"CSV Columns: {', '.join(headers)}\n"
            f"Total Data Rows: {len(data_rows)}\n"
            f"Sample Rows:\n" + "\n".join(sample_lines)
        )
        return summary_text, headers, len(data_rows), None

    @staticmethod
    def _read_docx(path: Path) -> tuple[str, str | None]:
        try:
            from docx import Document

            doc = Document(str(path))
            parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            return "\n".join(parts), None
        except ImportError:
            # Pure-stdlib XML fallback if python-docx is unavailable
            try:
                with zipfile.ZipFile(path) as archive:
                    xml_bytes = archive.read("word/document.xml")
                root = ET.fromstring(xml_bytes)
                paragraphs: list[str] = []
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                for para in root.iter(f"{ns}p"):
                    texts = [node.text for node in para.iter(f"{ns}t") if node.text]
                    line = "".join(texts).strip()
                    if line:
                        paragraphs.append(line)
                return "\n".join(paragraphs), None
            except Exception as exc:
                return "", f"Failed to parse DOCX '{path.name}': {exc}"
        except Exception as exc:
            return "", f"Failed to parse DOCX '{path.name}': {exc}"

    @staticmethod
    def _read_pdf(path: Path) -> tuple[str, int | None, str | None]:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages_text: list[str] = []
            for idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages_text.append(f"[Page {idx}]\n{extracted.strip()}")
            return "\n\n".join(pages_text), len(reader.pages), None
        except ImportError:
            return "", None, f"pypdf is not installed; cannot read PDF '{path.name}'."
        except Exception as exc:
            return "", None, f"Failed to parse PDF '{path.name}': {exc}"

    @staticmethod
    def _extract_headings(text: str, columns: list[str]) -> list[str]:
        headings: list[str] = []
        if columns:
            headings.extend(columns[:10])
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 80:
                continue
            if stripped.startswith("#"):
                headings.append(stripped.lstrip("# ").strip())
            elif stripped.endswith(":") and len(stripped.split()) <= 8:
                headings.append(stripped.rstrip(":"))
            elif stripped.isupper() and len(stripped.split()) <= 8:
                headings.append(stripped.title())
        return DocumentAnalyzer._dedupe_preserve_order(headings)[:15]

    @staticmethod
    def _infer_source_label(stem: str, headings: list[str]) -> str:
        humanized = re.sub(r"[_-]+", " ", stem).strip()
        if humanized:
            # Preserve uppercase acronyms like HR, IT, SLA
            words = [w.upper() if w.lower() in {"hr", "it", "sla", "pto", "faq"} else w for w in humanized.split()]
            if words:
                words[0] = words[0].upper() if words[0].lower() in {"hr", "it", "sla", "pto", "faq"} else words[0]
            return " ".join(words)
        if headings:
            return headings[0]
        return stem

    @staticmethod
    def _extract_key_topics(text: str, headings: list[str], columns: list[str]) -> list[str]:
        topics: list[str] = []
        for h in headings:
            if not h.lower().startswith("[page "):
                topics.append(h)
        for col in columns:
            topics.append(col.replace("_", " "))

        freq: dict[str, int] = {}
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text.lower()):
            if token in _COMMON_STOPWORDS:
                continue
            freq[token] = freq.get(token, 0) + 1
        top_terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:8]
        for term, _ in top_terms:
            topics.append(term)
        return DocumentAnalyzer._dedupe_preserve_order(topics)[:12]

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            norm = item.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(item.strip())
        return result

