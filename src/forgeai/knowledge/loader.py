"""Multi-file document loader supporting PDF, DOCX, TXT, CSV, and Markdown."""

from __future__ import annotations

import hashlib
from pathlib import Path

from forgeai.knowledge.schemas import DocumentType, LoadedFile

SUPPORTED_DOC_EXTENSIONS: dict[str, DocumentType] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".csv": "csv",
    ".md": "md",
    ".markdown": "md",
}


class DocumentLoader:
    """Discover and fingerprint supported knowledge documents inside a folder."""

    def load_folder(
        self,
        folder_path: str | Path,
        include_subdirectories: bool = True,
    ) -> tuple[list[LoadedFile], list[str]]:
        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(f"Documents folder does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected a directory for documents, got: {root}")

        pattern = "**/*" if include_subdirectories else "*"
        all_files = sorted(p for p in root.glob(pattern) if p.is_file())

        loaded: list[LoadedFile] = []
        skipped: list[str] = []

        for path in all_files:
            rel = str(path.relative_to(root)).replace("\\", "/")
            ext = path.suffix.lower()
            doc_type = SUPPORTED_DOC_EXTENSIONS.get(ext)
            if doc_type is None:
                skipped.append(rel)
                continue

            file_bytes = path.read_bytes()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
            loaded.append(
                LoadedFile(
                    document_name=path.name,
                    relative_path=rel,
                    absolute_path=str(path.resolve()),
                    document_type=doc_type,
                    size_bytes=len(file_bytes),
                    sha256=sha256,
                )
            )

        return loaded, skipped

