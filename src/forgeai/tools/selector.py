"""Dynamic Tool Selector with explicit per-tool decision reasons."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from forgeai.groundability.schemas import SupportedTopicConfig
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.schemas import KnowledgeManifest
from forgeai.tools.registry import ToolRegistry, build_default_registry
from forgeai.tools.schemas import (
    LLMToolSelectionBatch,
    ToolDecision,
    ToolRegistryEntry,
    ToolSelectionRecord,
)


class DynamicToolSelector:
    """Dynamically attach only the tools justified by requirements, groundability, and documents."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def select(
        self,
        requirements: AgentRequirements | str | Path,
        topic_config: SupportedTopicConfig | str | Path | None = None,
        manifest: KnowledgeManifest | str | Path | None = None,
        manifest_path: str | Path | None = None,
        top_k: int = 4,
        use_llm: bool | None = None,
        model: str = "gpt-4o-mini",
        output_dir: str | Path | None = None,
    ) -> ToolSelectionRecord:
        req = self._load_requirements(requirements)
        topic_cfg = self._load_topic_config(topic_config) if topic_config is not None else None
        manifest_obj, resolved_manifest_path = self._load_manifest(manifest, manifest_path)

        decisions = self._select_deterministic(
            req=req,
            topic_cfg=topic_cfg,
            manifest=manifest_obj,
            manifest_path=resolved_manifest_path,
            top_k=top_k,
        )

        should_use_llm = has_configured_llm_credentials(model) if use_llm is None else use_llm
        if should_use_llm:
            decisions = self._refine_with_llm(
                decisions=decisions,
                req=req,
                topic_cfg=topic_cfg,
                manifest=manifest_obj,
                model=model,
            )

        selected_tools = [d.tool_id for d in decisions if d.selected]
        rejected_tools = [d.tool_id for d in decisions if not d.selected]
        formatted_summary = "\n".join(d.format_line() for d in decisions)

        record = ToolSelectionRecord(
            agent_name=topic_cfg.agent_name if topic_cfg else req.agent_name,
            selected_tools=selected_tools,
            rejected_tools=rejected_tools,
            decisions=decisions,
            formatted_summary=formatted_summary,
            metadata={
                "selection_mode": "llm_assisted" if should_use_llm else "deterministic",
                "total_registered_tools": len(decisions),
                "selected_count": len(selected_tools),
            },
        )

        if output_dir is not None:
            self.save_outputs(record, output_dir)

        return record

    def save_outputs(self, record: ToolSelectionRecord, output_dir: str | Path) -> dict[str, Path]:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        registry_path = self.registry.save_json(out_root / "tool_registry.json")
        selection_path = out_root / "tool_selection.json"
        selection_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return {"tool_registry": registry_path, "tool_selection": selection_path}

    def _select_deterministic(
        self,
        req: AgentRequirements,
        topic_cfg: SupportedTopicConfig | None,
        manifest: KnowledgeManifest | None,
        manifest_path: str | None,
        top_k: int,
    ) -> list[ToolDecision]:
        restrictions = topic_cfg.restrictions if topic_cfg else req.restrictions
        supported_topics = topic_cfg.supported_topics if topic_cfg else req.supported_topics
        purpose = topic_cfg.purpose if topic_cfg else req.purpose
        context_text = " ".join(
            [
                req.agent_name,
                req.domain,
                purpose,
                " ".join(supported_topics),
                " ".join(req.sources),
            ]
        ).lower()
        restrictions_lower = " ".join(restrictions).lower()

        closed_book = any(
            phrase in restrictions_lower
            for phrase in (
                "approved knowledge base",
                "strictly from provided",
                "strictly in the verified source",
                "no invented",
                "do not invent",
            )
        )

        csv_docs = (
            [d for d in manifest.documents if d.document_type == "csv"]
            if manifest is not None
            else []
        )
        csv_paths = (
            [str((Path(manifest.docs_dir) / d.relative_path).resolve()) for d in csv_docs]
            if manifest is not None and manifest.docs_dir
            else []
        )

        decisions: list[ToolDecision] = []
        for entry in self.registry.list_entries():
            decisions.append(
                self._evaluate_entry(
                    entry=entry,
                    req=req,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    top_k=top_k,
                    csv_docs=csv_docs,
                    csv_paths=csv_paths,
                    context_text=context_text,
                    closed_book=closed_book,
                )
            )
        return decisions

    @staticmethod
    def _evaluate_entry(
        entry: ToolRegistryEntry,
        req: AgentRequirements,
        manifest: KnowledgeManifest | None,
        manifest_path: str | None,
        top_k: int,
        csv_docs: list[Any],
        csv_paths: list[str],
        context_text: str,
        closed_book: bool,
    ) -> ToolDecision:
        tid = entry.tool_id

        if tid == "document_retriever":
            has_kb = (manifest is not None and manifest.total_chunks > 0) or bool(manifest_path)
            if has_kb:
                doc_count = manifest.total_documents if manifest else len(req.sources)
                chunk_count = manifest.total_chunks if manifest else 0
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason=(
                        f"Knowledge base contains {chunk_count} indexed chunk(s) across {doc_count} uploaded "
                        "document(s) and requires semantic retrieval for grounded answers."
                    ),
                    parameters={"manifest_path": manifest_path, "top_k": top_k},
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason="No ingested knowledge base or manifest was provided.",
            )

        if tid == "csv_lookup":
            if csv_docs:
                names = ", ".join(d.document_name for d in csv_docs)
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason=(
                        f"Uploaded structured CSV data ({names}) is present and supports exact "
                        "tabular record lookups."
                    ),
                    parameters={"csv_paths": csv_paths},
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason="No CSV tables or catalogs were uploaded in the document set.",
            )

        if tid == "calculator":
            calc_cues = (
                "calculat",
                "arithmetic",
                "cost",
                "price",
                "pricing",
                "budget",
                "total",
                "quantity",
                "percent",
                "stipend",
                "quota",
                "allowance",
                "procurement",
                "inventory",
                "invoice",
                "leave",
                "pto",
                "days",
                "hours",
                "carry over",
                "prorate",
            )
            doc_text = " ".join(c.content.lower() for c in manifest.chunks) if manifest else ""
            if any(cue in context_text or cue in doc_text for cue in calc_cues):
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason=(
                        "Domain and documents include numeric quantities, allowances, limits, or pricing "
                        "where deterministic arithmetic prevents LLM math errors."
                    ),
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason="Requirements and documents are purely qualitative with no arithmetic needs.",
            )

        if tid == "date_tool":
            date_cues = (
                "leave",
                "pto",
                "attendance",
                "holiday",
                "schedule",
                "deadline",
                "calendar",
                "tenure",
                "enrollment",
                "business day",
                "notice",
                "date",
                "expir",
                "q1",
                "hr",
                "human resources",
            )
            doc_text = " ".join(c.content.lower() for c in manifest.chunks) if manifest else ""
            if any(cue in context_text or cue in doc_text for cue in date_cues):
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason=(
                        "Policy rules involve timelines, notice periods, holidays, or eligibility windows "
                        "that benefit from accurate date and business-day calculations."
                    ),
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason="Use case does not involve schedules, deadlines, or date arithmetic.",
            )

        if tid == "http_api":
            wants_api = bool(re.search(r"\b(rest\s+api|http\s+api|endpoint|webhook|live\s+api|external\s+api)\b", context_text))
            if wants_api and not closed_book:
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason="Requirements explicitly call for querying an external HTTP/REST API.",
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason=(
                    "Excluded because agent operates strictly on internal approved documents without external API calls."
                    if closed_book
                    else "No external HTTP/REST API integration was requested."
                ),
            )

        if tid == "web_search":
            wants_web = bool(re.search(r"\b(web\s+search|internet|latest\s+news|online\s+search|google|browse)\b", context_text))
            if wants_web and not closed_book:
                return ToolDecision(
                    tool_id=tid,
                    display_name=entry.display_name,
                    priority=entry.priority,
                    selected=True,
                    reason="Requirements request external web information and closed-book constraints do not apply.",
                )
            return ToolDecision(
                tool_id=tid,
                display_name=entry.display_name,
                priority=entry.priority,
                selected=False,
                reason=(
                    "Excluded because closed-book restrictions forbid unapproved external web sources."
                    if closed_book
                    else "External web search was not requested in the brief."
                ),
            )

        # Generic custom tool default
        return ToolDecision(
            tool_id=tid,
            display_name=entry.display_name,
            priority=entry.priority,
            selected=False,
            reason="Custom tool not explicitly matched to current requirements.",
        )

    def _refine_with_llm(
        self,
        decisions: list[ToolDecision],
        req: AgentRequirements,
        topic_cfg: SupportedTopicConfig | None,
        manifest: KnowledgeManifest | None,
        model: str,
    ) -> list[ToolDecision]:
        restrictions = topic_cfg.restrictions if topic_cfg else req.restrictions
        closed_book = any("strictly" in r.lower() or "no invented" in r.lower() for r in restrictions)
        prompt = (
            "Review and refine tool selection decisions for this AI agent. "
            "Every tool MUST have a concise, concrete reason.\n"
            f"Agent: {req.agent_name} ({req.domain})\n"
            f"Purpose: {topic_cfg.purpose if topic_cfg else req.purpose}\n"
            f"Documents: {[d.document_name for d in manifest.documents] if manifest else req.sources}\n"
            f"Restrictions: {restrictions}\n\n"
            "Current candidate decisions:\n"
            + "\n".join(d.format_line() for d in decisions)
        )
        batch = complete_structured(prompt=prompt, response_model=LLMToolSelectionBatch, model=model, temperature=0.0)
        by_id = {d.tool_id: d for d in batch.decisions}

        refined: list[ToolDecision] = []
        for orig in decisions:
            entry = self.registry.get_entry(orig.tool_id)
            cand = by_id.get(orig.tool_id)
            if cand is None:
                refined.append(orig)
                continue
            # Hard safety guard: never allow unsafe external network tools under closed-book restrictions
            selected = cand.selected
            reason = cand.reason
            if closed_book and not entry.safe_for_closed_book and selected:
                selected = False
                reason = "Blocked by closed-book restriction: only approved internal documents may be used."
            refined.append(
                orig.model_copy(update={"selected": selected, "reason": reason})
            )
        return refined

    @staticmethod
    def _load_requirements(requirements: AgentRequirements | str | Path) -> AgentRequirements:
        if isinstance(requirements, AgentRequirements):
            return requirements
        return AgentRequirements.model_validate_json(Path(requirements).read_text(encoding="utf-8"))

    @staticmethod
    def _load_topic_config(cfg: SupportedTopicConfig | str | Path) -> SupportedTopicConfig:
        if isinstance(cfg, SupportedTopicConfig):
            return cfg
        return SupportedTopicConfig.model_validate_json(Path(cfg).read_text(encoding="utf-8"))

    @staticmethod
    def _load_manifest(
        manifest: KnowledgeManifest | str | Path | None,
        manifest_path: str | Path | None,
    ) -> tuple[KnowledgeManifest | None, str | None]:
        if manifest is None:
            if manifest_path is not None:
                p = Path(manifest_path)
                return KnowledgeManifest.model_validate_json(p.read_text(encoding="utf-8")), str(p.resolve())
            return None, None
        if isinstance(manifest, KnowledgeManifest):
            resolved = str(Path(manifest_path).resolve()) if manifest_path else None
            return manifest, resolved
        p = Path(manifest)
        return KnowledgeManifest.model_validate_json(p.read_text(encoding="utf-8")), str(p.resolve())

