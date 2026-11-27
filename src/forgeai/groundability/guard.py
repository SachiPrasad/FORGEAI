"""Feature 3 — Groundability Guard: requirement-to-document mapping & topic coverage analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from forgeai.groundability.schemas import (
    CoverageStatus,
    DocumentSupportLabel,
    GroundabilityReport,
    LLMGroundabilityBatch,
    SupportedTopicConfig,
    TopicAssessment,
)
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.retriever import KnowledgeRetriever
from forgeai.knowledge.schemas import KnowledgeManifest, RetrievalResult

_GENERIC_TOPIC_WORDS = {
    "policy",
    "policies",
    "rule",
    "rules",
    "benefit",
    "benefits",
    "employee",
    "employees",
    "company",
    "information",
    "details",
    "overview",
    "guide",
    "handbook",
    "management",
    "program",
    "plan",
    "plans",
}

_PARTIAL_QUALIFIER_PATTERNS = (
    re.compile(r"\b(?:only|basic|brief|summary|limited|contact\s+hr|not\s+detailed|ask\s+hr|refer\s+to)\b", re.I),
    re.compile(r"\b(?:separate\s+portal|external\s+provider|enrollment\s+only|high[- ]level)\b", re.I),
)


class GroundabilityGuard:
    """Verify whether requested agent topics are genuinely supported by the ingested knowledge base."""

    def evaluate(
        self,
        requirements: AgentRequirements | str | Path,
        manifest: KnowledgeManifest | str | Path,
        extra_topics: list[str] | None = None,
        top_k: int = 5,
        allow_partial_topics: bool = True,
        use_llm: bool | None = None,
        model: str = "gpt-4o-mini",
        output_dir: str | Path | None = None,
    ) -> tuple[GroundabilityReport, SupportedTopicConfig]:
        """Run requirement-to-document mapping, semantic comparison, and restriction generation."""
        req = self._load_requirements(requirements)
        manifest_obj, manifest_path = self._load_manifest(manifest)

        if manifest_path is not None:
            retriever = KnowledgeRetriever.from_manifest(manifest_path)
        else:
            raise ValueError("manifest must be a path or a KnowledgeManifest resolved from disk.")

        topics_to_check = self._collect_topics(req, extra_topics=extra_topics)
        available_docs = [doc.document_name for doc in manifest_obj.documents]

        # Retrieve evidence for each topic and score deterministically
        raw_evidence: dict[str, list[RetrievalResult]] = {}
        assessments: list[TopicAssessment] = []
        for topic in topics_to_check:
            hits = retriever.retrieve(
                query=topic,
                top_k=top_k,
                min_score=-1.0,
                hybrid_keyword_weight=0.35,
            )
            raw_evidence[topic] = hits
            assessments.append(self._assess_topic_deterministic(topic, hits, manifest_obj))

        # Optional LLM groundability verification when explicitly enabled or credentials exist
        should_use_llm = has_configured_llm_credentials(model) if use_llm is None else use_llm
        llm_used = False
        if should_use_llm and assessments:
            assessments = self._refine_with_llm(assessments, raw_evidence, model=model)
            llm_used = True

        supported = [a.topic for a in assessments if a.status == "supported"]
        partial = [a.topic for a in assessments if a.status == "partial"]
        unsupported = [a.topic for a in assessments if a.status == "unsupported"]

        # Merge any already-flagged unsupported topics from intake that aren't already represented
        existing_lower = {t.lower() for t in (supported + partial + unsupported)}
        for pre_unsupported in req.unsupported_topics:
            if pre_unsupported.strip().lower() not in existing_lower:
                unsupported.append(pre_unsupported)

        groundability_score = self._compute_groundability_score(assessments)
        generated_restrictions = self._generate_restrictions(
            existing_restrictions=req.restrictions,
            supported=supported,
            partial=partial,
            unsupported=unsupported,
            available_docs=available_docs,
        )
        recommendation = self._build_recommendation(supported, partial, unsupported)
        refusal_policy = self._build_refusal_policy(supported, partial, unsupported, available_docs)

        report = GroundabilityReport(
            agent_name=req.agent_name,
            domain=req.domain,
            total_topics_evaluated=len(assessments),
            groundability_score=groundability_score,
            supported_topics=supported,
            partially_supported_topics=partial,
            unsupported_topics=unsupported,
            topic_assessments=assessments,
            generated_restrictions=generated_restrictions,
            recommendation=recommendation,
            available_documents=available_docs,
            metadata={
                "verification_mode": "llm_assisted" if llm_used else "semantic_heuristic",
                "model": model if llm_used else None,
                "total_chunks_searched": manifest_obj.total_chunks,
            },
        )
        report.formatted_report = self.format_report(report)

        partial_caveats = {
            a.topic: (
                f"Only partially covered in {', '.join(a.supporting_documents) or 'available documents'}. "
                f"{a.reason} Do not extrapolate missing details."
            )
            for a in assessments
            if a.status == "partial"
        }

        supported_config = SupportedTopicConfig(
            agent_name=req.agent_name,
            domain=req.domain,
            target_audience=req.target_audience,
            purpose=self._scope_purpose(req.purpose, supported, partial, allow_partial_topics),
            tone=req.tone,
            output_format=req.output_format,
            supported_topics=supported,
            partially_supported_topics=partial if allow_partial_topics else [],
            unsupported_topics=unsupported if allow_partial_topics else [*unsupported, *partial],
            allow_partial_topics=allow_partial_topics,
            partial_topic_caveats=partial_caveats,
            restrictions=generated_restrictions,
            refusal_policy=refusal_policy,
            allowed_sources=available_docs,
            groundability_score=groundability_score,
            recommendation=recommendation,
        )

        if output_dir is not None:
            self.save_outputs(report, supported_config, output_dir)

        return report, supported_config

    @staticmethod
    def save_outputs(
        report: GroundabilityReport,
        supported_config: SupportedTopicConfig,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        report_json = out_root / "groundability_report.json"
        report_md = out_root / "groundability_report.md"
        config_json = out_root / "supported_topic_config.json"

        report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report_md.write_text(report.formatted_report, encoding="utf-8")
        config_json.write_text(supported_config.model_dump_json(indent=2), encoding="utf-8")

        return {
            "report_json": report_json,
            "report_md": report_md,
            "supported_topic_config": config_json,
        }

    @staticmethod
    def format_report(report: GroundabilityReport) -> str:
        lines: list[str] = ["GROUNDABILITY REPORT", ""]

        lines.append("Supported:")
        if report.supported_topics:
            lines.extend(f"- {t}" for t in report.supported_topics)
        else:
            lines.append("- None")
        lines.append("")

        lines.append("Partially Supported:")
        if report.partially_supported_topics:
            lines.extend(f"- {t}" for t in report.partially_supported_topics)
        else:
            lines.append("- None")
        lines.append("")

        lines.append("Unsupported:")
        if report.unsupported_topics:
            lines.extend(f"- {t}" for t in report.unsupported_topics)
        else:
            lines.append("- None")
        lines.append("")

        lines.append("Topic Coverage Table:")
        lines.append("| Requested topic | Document support | Status | Score | Sources |")
        lines.append("| --- | --- | --- | --- | --- |")
        for a in report.topic_assessments:
            sources = ", ".join(a.supporting_documents) if a.supporting_documents else "-"
            lines.append(
                f"| {a.topic} | {a.document_support} | {a.status.title()} | {a.semantic_score:.2f} | {sources} |"
            )
        lines.append("")
        lines.append("Recommendation:")
        lines.append(report.recommendation)
        return "\n".join(lines)

    def _assess_topic_deterministic(
        self,
        topic: str,
        hits: list[RetrievalResult],
        manifest: KnowledgeManifest,
    ) -> TopicAssessment:
        topic_tokens = self._topic_tokens(topic)
        distinctive_tokens = [t for t in topic_tokens if t not in _GENERIC_TOPIC_WORDS]
        if not distinctive_tokens:
            distinctive_tokens = topic_tokens

        # Find chunks that actually contain at least one distinctive topic token (or stem prefix)
        matched_hits: list[tuple[RetrievalResult, float]] = []
        for hit in hits:
            haystack = f"{hit.chunk.document_name} {hit.chunk.section or ''} {hit.chunk.content}".lower()
            cov = self._token_coverage(distinctive_tokens, haystack)
            if cov > 0.0:
                matched_hits.append((hit, cov))

        if not matched_hits:
            top_sem = max((max(0.0, h.semantic_score) for h in hits), default=0.0)
            return TopicAssessment(
                topic=topic,
                document_support="Missing",
                status="unsupported",
                semantic_score=round(top_sem, 4),
                keyword_coverage=0.0,
                evidence_count=0,
                supporting_documents=[],
                evidence_citations=[],
                evidence_snippets=[],
                reason=f"No ingested document contains substantive coverage for '{topic}'.",
            )

        best_hit, best_cov = max(matched_hits, key=lambda pair: (pair[1], pair[0].score))
        best_sem = max(0.0, best_hit.semantic_score)
        supporting_docs = self._dedupe([h.chunk.document_name for h, _ in matched_hits])
        citations = self._dedupe([h.citation for h, _ in matched_hits[:3]])
        snippets = [h.chunk.content[:220] for h, _ in matched_hits[:2]]

        combined_text = " ".join(h.chunk.content for h, _ in matched_hits)
        matching_sentences = self._matching_sentences(combined_text, distinctive_tokens)
        matching_sentence_words = sum(len(s.split()) for s in matching_sentences)
        has_partial_caveat = any(pat.search(s) for s in matching_sentences for pat in _PARTIAL_QUALIFIER_PATTERNS)

        # Check if a dedicated document or section title matches the distinctive tokens
        has_dedicated_doc_or_section = any(
            self._token_coverage(distinctive_tokens, f"{h.chunk.document_name} {h.chunk.section or ''}".lower()) >= 0.5
            for h, _ in matched_hits
        )

        # Classification decision:
        # - Supported: strong distinctive concept coverage + substantive multi-sentence or dedicated doc/section detail
        # - Partial: shallow/single-sentence mention, partial caveat, or only subset of multi-word concept matched
        # - Unsupported: insufficient match
        if best_cov >= 0.99 and not has_partial_caveat and (
            has_dedicated_doc_or_section or len(matched_hits) >= 2 or len(matching_sentences) >= 2 or matching_sentence_words >= 30
        ):
            status: CoverageStatus = "supported"
            doc_support: DocumentSupportLabel = "Available"
            reason = (
                f"Substantively supported by {', '.join(supporting_docs)} "
                f"({len(matched_hits)} matching chunk(s), {len(matching_sentences)} relevant sentence(s))."
            )
        elif best_cov >= 0.5 and len(matched_hits) >= 1:
            status = "partial"
            doc_support = "Partial"
            if has_partial_caveat:
                reason = (
                    f"Only partially covered in {', '.join(supporting_docs)}; "
                    "documents contain high-level or referral language without full details."
                )
            else:
                reason = (
                    f"Partially covered in {', '.join(supporting_docs)}, but lacks comprehensive "
                    "dedicated policy depth."
                )
        else:
            status = "unsupported"
            doc_support = "Missing"
            reason = f"Available documents do not sufficiently cover '{topic}'."

        return TopicAssessment(
            topic=topic,
            document_support=doc_support,
            status=status,
            semantic_score=round(best_sem, 4),
            keyword_coverage=round(best_cov, 4),
            evidence_count=len(matched_hits),
            supporting_documents=supporting_docs if status != "unsupported" else [],
            evidence_citations=citations if status != "unsupported" else [],
            evidence_snippets=snippets if status != "unsupported" else [],
            reason=reason,
        )

    def _refine_with_llm(
        self,
        assessments: list[TopicAssessment],
        raw_evidence: dict[str, list[RetrievalResult]],
        model: str,
    ) -> list[TopicAssessment]:
        evidence_blocks: list[str] = []
        for a in assessments:
            hits = raw_evidence.get(a.topic, [])[:3]
            if hits:
                formatted = "\n".join(f"  - [{h.citation}] {h.chunk.content[:300]}" for h in hits)
            else:
                formatted = "  - No retrieved chunks."
            evidence_blocks.append(f"Topic: {a.topic}\nHeuristic status: {a.status}\nRetrieved evidence:\n{formatted}")

        prompt = (
            "You are a strict Groundability Guard auditor. Classify each requested topic as:\n"
            "- 'supported': documents provide clear, substantive rules/information to answer user questions.\n"
            "- 'partial': documents only mention it briefly, lack detailed rules, or explicitly defer elsewhere.\n"
            "- 'unsupported': documents do not contain information for this topic.\n\n"
            "Never mark a topic supported just because a generic word like 'policy' or 'benefits' appears.\n\n"
            + "\n\n".join(evidence_blocks)
        )
        batch = complete_structured(prompt=prompt, response_model=LLMGroundabilityBatch, model=model, temperature=0.0)
        verdict_map = {v.topic.strip().lower(): v for v in batch.verdicts}

        updated: list[TopicAssessment] = []
        support_label_map: dict[CoverageStatus, DocumentSupportLabel] = {
            "supported": "Available",
            "partial": "Partial",
            "unsupported": "Missing",
        }
        for a in assessments:
            verdict = verdict_map.get(a.topic.strip().lower())
            if verdict:
                updated.append(
                    a.model_copy(
                        update={
                            "status": verdict.status,
                            "document_support": support_label_map[verdict.status],
                            "reason": verdict.reason,
                            "supporting_documents": [] if verdict.status == "unsupported" else a.supporting_documents,
                            "evidence_citations": [] if verdict.status == "unsupported" else a.evidence_citations,
                            "evidence_snippets": [] if verdict.status == "unsupported" else a.evidence_snippets,
                        }
                    )
                )
            else:
                updated.append(a)
        return updated

    @staticmethod
    def _load_requirements(requirements: AgentRequirements | str | Path) -> AgentRequirements:
        if isinstance(requirements, AgentRequirements):
            return requirements
        return AgentRequirements.model_validate_json(Path(requirements).read_text(encoding="utf-8"))

    @staticmethod
    def _load_manifest(manifest: KnowledgeManifest | str | Path) -> tuple[KnowledgeManifest, Path | None]:
        if isinstance(manifest, KnowledgeManifest):
            return manifest, None
        path = Path(manifest)
        return KnowledgeManifest.model_validate_json(path.read_text(encoding="utf-8")), path

    @classmethod
    def _collect_topics(cls, req: AgentRequirements, extra_topics: list[str] | None) -> list[str]:
        candidates = list(req.supported_topics)
        if extra_topics:
            candidates.extend(extra_topics)
        return cls._dedupe(candidates)

    @staticmethod
    def _topic_tokens(topic: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", topic.lower())
        normalized: list[str] = []
        for tok in tokens:
            if tok.endswith("ies") and len(tok) > 4:
                normalized.append(tok[:-3] + "y")
            elif tok.endswith("s") and len(tok) > 4 and not tok.endswith("ss"):
                normalized.append(tok[:-1])
            else:
                normalized.append(tok)
        return normalized

    @staticmethod
    def _token_coverage(tokens: list[str], haystack: str) -> float:
        if not tokens:
            return 0.0
        matched = 0
        for tok in tokens:
            stem = tok[:5] if len(tok) > 5 else tok
            if re.search(rf"\b{re.escape(tok)}", haystack) or (len(stem) >= 4 and stem in haystack):
                matched += 1
        return matched / len(tokens)

    @staticmethod
    def _compute_groundability_score(assessments: list[TopicAssessment]) -> float:
        if not assessments:
            return 0.0
        weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
        total = sum(weights[a.status] for a in assessments)
        return round(total / len(assessments), 2)

    @classmethod
    def _generate_restrictions(
        cls,
        existing_restrictions: list[str],
        supported: list[str],
        partial: list[str],
        unsupported: list[str],
        available_docs: list[str],
    ) -> list[str]:
        restrictions = list(existing_restrictions)
        restrictions.append(
            "Do not invent information or pretend that missing documents exist."
        )
        if available_docs:
            restrictions.append(
                f"Ground all factual answers strictly in the verified source documents: {', '.join(available_docs)}."
            )
        if unsupported:
            restrictions.append(
                f"Explicitly refuse requests about unsupported topics ({', '.join(unsupported)}) "
                "and inform the user that supporting documentation is unavailable."
            )
        if partial:
            restrictions.append(
                f"For partially supported topics ({', '.join(partial)}), share only explicitly documented facts "
                "and clearly state any missing details."
            )
        if supported:
            restrictions.append(
                f"Limit primary topic coverage to verified supported topics: {', '.join(supported)}."
            )
        return cls._dedupe(restrictions)

    @staticmethod
    def _build_recommendation(supported: list[str], partial: list[str], unsupported: list[str]) -> str:
        if not supported and not partial:
            return (
                "Block agent generation until supporting documents are uploaded. "
                "None of the requested topics are grounded in the current knowledge base."
            )
        if unsupported or partial:
            return (
                "Generate the agent with supported topics only. "
                "Unsupported topics must be explicitly identified and refused."
            )
        return "All requested topics are fully grounded in the knowledge base. Safe to generate the agent."

    @staticmethod
    def _build_refusal_policy(
        supported: list[str],
        partial: list[str],
        unsupported: list[str],
        available_docs: list[str],
    ) -> str:
        docs_text = ", ".join(available_docs) if available_docs else "none"
        unsupported_text = ", ".join(unsupported) if unsupported else "any topic outside the uploaded documents"
        partial_text = f" For partially covered topics ({', '.join(partial)}), state caveats clearly." if partial else ""
        return (
            f"If a user asks about {unsupported_text}, do NOT guess or fabricate an answer. "
            f"Respond politely that this topic is not covered by the available documents ({docs_text}), "
            f"list what you can help with ({', '.join(supported) if supported else 'supported policy topics'}), "
            f"and direct them to the appropriate team.{partial_text}"
        )

    @staticmethod
    def _scope_purpose(
        original_purpose: str,
        supported: list[str],
        partial: list[str],
        allow_partial_topics: bool,
    ) -> str:
        scoped = list(supported)
        if allow_partial_topics:
            scoped.extend(partial)
        if not scoped:
            return original_purpose
        return f"{original_purpose.rstrip('.')} (scoped to verified topics: {', '.join(scoped)})."

    @staticmethod
    def _matching_sentences(text: str, tokens: list[str]) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        return [
            s for s in sentences
            if any(
                re.search(rf"\b{re.escape(tok)}", s.lower()) or (len(tok) > 5 and tok[:5] in s.lower())
                for tok in tokens
            )
        ]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            norm = item.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(item.strip())
        return out
