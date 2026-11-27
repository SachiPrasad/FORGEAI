"""Knowledge base validation: empty files, duplicate hashes, corrupted files, and version conflicts."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.knowledge.schemas import IngestedDocumentInfo, LoadedFile


class KnowledgeValidationIssue(BaseModel):
    """Single validation warning/error detected during knowledge ingestion."""

    severity: str
    issue_type: str
    document_name: str
    message: str


class KnowledgeValidationReport(BaseModel):
    """Aggregated validation summary stored in knowledge_manifest.json."""

    is_healthy: bool = True
    empty_files: list[str] = Field(default_factory=list)
    duplicate_files: list[str] = Field(default_factory=list)
    corrupted_files: list[str] = Field(default_factory=list)
    version_conflicts: list[str] = Field(default_factory=list)
    issues: list[KnowledgeValidationIssue] = Field(default_factory=list)


class KnowledgeBaseValidator:
    """Inspect loaded & extracted documents for duplicates, empty files, corruption, and version collisions."""

    def validate(
        self,
        loaded_files: list[LoadedFile],
        extracted_docs: list[IngestedDocumentInfo],
    ) -> KnowledgeValidationReport:
        empty_files: list[str] = []
        duplicate_files: list[str] = []
        corrupted_files: list[str] = []
        version_conflicts: list[str] = []
        issues: list[KnowledgeValidationIssue] = []

        # 1. Duplicate content hash detection
        seen_hashes: dict[str, str] = {}
        # 2. Versioned stem collision detection (e.g., policy_v1.pdf vs policy_v2.pdf)
        base_stems: dict[str, list[str]] = {}

        for f in loaded_files:
            if f.size_bytes == 0:
                empty_files.append(f.relative_path)
                issues.append(
                    KnowledgeValidationIssue(
                        severity="warning",
                        issue_type="empty_file",
                        document_name=f.relative_path,
                        message=f"File '{f.relative_path}' is 0 bytes.",
                    )
                )

            if f.sha256 in seen_hashes:
                duplicate_files.append(f.relative_path)
                issues.append(
                    KnowledgeValidationIssue(
                        severity="warning",
                        issue_type="duplicate_content",
                        document_name=f.relative_path,
                        message=f"'{f.relative_path}' has identical content to '{seen_hashes[f.sha256]}'.",
                    )
                )
            else:
                seen_hashes[f.sha256] = f.relative_path

            stem = Path(f.document_name).stem.lower()
            normalized_stem = re.sub(r"[_\-\s]*v?\d+(?:\.\d+)*$", "", stem).strip("_- ")
            if normalized_stem:
                base_stems.setdefault(normalized_stem, []).append(f.relative_path)

        for norm, paths in base_stems.items():
            # Only flag if multiple files share the stripped stem and at least one has a version suffix
            if len(paths) > 1 and any(re.search(r"[_\-\s]*v?\d+", Path(p).stem.lower()) for p in paths):
                conflict_label = f"{norm}: {', '.join(paths)}"
                version_conflicts.append(conflict_label)
                issues.append(
                    KnowledgeValidationIssue(
                        severity="warning",
                        issue_type="version_conflict",
                        document_name=paths[-1],
                        message=f"Possible conflicting document versions detected: {conflict_label}.",
                    )
                )

        # 3. Extraction empty / corrupted file checks
        for doc in extracted_docs:
            warn_text = " ".join(doc.warnings).lower()
            if "failed to" in warn_text or "could not decode" in warn_text:
                corrupted_files.append(doc.file.relative_path)
                issues.append(
                    KnowledgeValidationIssue(
                        severity="error",
                        issue_type="corrupted_file",
                        document_name=doc.file.relative_path,
                        message="; ".join(doc.warnings),
                    )
                )
            elif not doc.segments and doc.file.relative_path not in empty_files:
                empty_files.append(doc.file.relative_path)
                issues.append(
                    KnowledgeValidationIssue(
                        severity="warning",
                        issue_type="empty_extracted_text",
                        document_name=doc.file.relative_path,
                        message=f"No text segments could be extracted from '{doc.file.relative_path}'.",
                    )
                )

        return KnowledgeValidationReport(
            is_healthy=len(corrupted_files) == 0,
            empty_files=empty_files,
            duplicate_files=duplicate_files,
            corrupted_files=corrupted_files,
            version_conflicts=version_conflicts,
            issues=issues,
        )

