"""Text cleaning and normalization utilities for extracted document segments."""

from __future__ import annotations

import re
import unicodedata

from forgeai.knowledge.schemas import ExtractedSegment


class TextCleaner:
    """Normalize Unicode, line endings, hyphenation, and excess whitespace."""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize Unicode compatibility characters (e.g., smart quotes, ligatures)
        normalized = unicodedata.normalize("NFKC", text)
        # Strip non-printable control characters except \n and \t
        normalized = "".join(ch for ch in normalized if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")
        # Standardize line endings
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        # Rejoin hyphenated line breaks commonly produced by PDF extraction
        normalized = re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", normalized)
        # Collapse spaces/tabs within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
        cleaned = "\n".join(lines)
        # Collapse 3+ consecutive newlines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def clean_segments(self, segments: list[ExtractedSegment]) -> list[ExtractedSegment]:
        cleaned_segments: list[ExtractedSegment] = []
        for seg in segments:
            cleaned = self.clean_text(seg.text)
            if not cleaned:
                continue
            cleaned_segments.append(seg.model_copy(update={"text": cleaned}))
        return cleaned_segments

