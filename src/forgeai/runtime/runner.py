"""Feature 10 — Generated Agent Runtime: loads agent.json and executes chat with tools, RAG, and memory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from forgeai.generator.schemas import AgentDefinition
from forgeai.intake.llm_client import has_configured_llm_credentials, resolve_effective_model
from forgeai.knowledge.retriever import KnowledgeRetriever
from forgeai.knowledge.schemas import RetrievalResult
from forgeai.runtime.memory import ConversationMemory
from forgeai.runtime.safety import detect_prompt_injection, is_tool_permitted
from forgeai.runtime.schemas import RuntimeReply, ToolCallTrace
from forgeai.tools.registry import ToolRegistry, build_default_registry


class _StructuredRuntimeLLMOutput(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    refused_unsupported_topic: bool = False


class AgentRuntime:
    """Runnable agent runtime loaded from agent.json."""

    def __init__(
        self,
        agent_def: AgentDefinition | str | Path,
        memory: ConversationMemory | None = None,
        registry: ToolRegistry | None = None,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self.workspace_dir: Path | None = (
            Path(workspace_dir)
            if workspace_dir is not None
            else (Path(agent_def).resolve().parent if isinstance(agent_def, (str, Path)) else None)
        )
        self.agent_def = self._load_agent(agent_def)
        self.memory = memory or ConversationMemory()
        self.registry = registry or build_default_registry()
        self.retriever: KnowledgeRetriever | None = None
        self.bound_tools: dict[str, Callable[..., Any]] = {}
        self._init_components()

    @classmethod
    def from_file(cls, agent_json_path: str | Path) -> AgentRuntime:
        return cls(agent_def=agent_json_path)

    def _resolve_portable_file(self, raw_path: str | None, fallbacks: list[Path]) -> str | None:
        """Resolve a saved artifact/document path across Windows/Linux environments."""
        if raw_path:
            direct = Path(raw_path)
            if direct.is_file():
                return str(direct)
            normalized = raw_path.replace("\\", "/")
            if "demos/" in normalized:
                rel_demo = normalized[normalized.index("demos/") :]
                for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
                    candidate = base / rel_demo
                    if candidate.is_file():
                        return str(candidate)
        for fb in fallbacks:
            if fb.is_file():
                return str(fb)
        return raw_path

    def _init_components(self) -> None:
        manifest_fallbacks: list[Path] = []
        if self.workspace_dir is not None:
            manifest_fallbacks.extend(
                [
                    self.workspace_dir / "kb" / "knowledge_manifest.json",
                    self.workspace_dir / "knowledge_manifest.json",
                ]
            )
        manifest_path = self._resolve_portable_file(
            self.agent_def.retrieval_config.knowledge_manifest_path,
            manifest_fallbacks,
        )
        if manifest_path and Path(manifest_path).is_file():
            self.retriever = KnowledgeRetriever.from_manifest(manifest_path)

        for spec in self.agent_def.tools:
            if not spec.enabled or not self.registry.has_tool(spec.name):
                continue
            try:
                kwargs = dict(spec.parameters)
                if spec.name == "document_retriever":
                    resolved_mp = self._resolve_portable_file(
                        kwargs.get("manifest_path") or manifest_path,
                        manifest_fallbacks,
                    )
                    if not resolved_mp or not Path(resolved_mp).is_file():
                        continue
                    kwargs["manifest_path"] = resolved_mp
                elif spec.name == "csv_lookup" and isinstance(kwargs.get("csv_paths"), list):
                    resolved_csvs: list[str] = []
                    for raw_csv in kwargs["csv_paths"]:
                        fname = re.split(r"[\\/]", str(raw_csv))[-1]
                        csv_fallbacks: list[Path] = []
                        if self.workspace_dir is not None and fname:
                            csv_fallbacks.extend(
                                [
                                    self.workspace_dir.parent / "docs" / fname,
                                    self.workspace_dir / "docs" / fname,
                                    self.workspace_dir / fname,
                                ]
                            )
                        resolved_csv = self._resolve_portable_file(str(raw_csv), csv_fallbacks)
                        if resolved_csv and Path(resolved_csv).is_file():
                            resolved_csvs.append(resolved_csv)
                    if resolved_csvs:
                        kwargs["csv_paths"] = resolved_csvs
                self.bound_tools[spec.name] = self.registry.build_callable(spec.name, **kwargs)
            except Exception:
                # Keep runtime resilient if an optional tool cannot bind
                continue

    def chat(self, user_message: str, use_llm: bool | None = None) -> RuntimeReply:
        """Process one user message, update conversation memory, and return cited RuntimeReply."""
        cleaned_msg = user_message.strip()
        if not cleaned_msg:
            return RuntimeReply(
                answer="Please enter a question so I can help.",
                confidence="low",
                error="Empty user message.",
            )

        self.memory.add_user_message(cleaned_msg)
        tool_traces: list[ToolCallTrace] = []

        injection_reason = detect_prompt_injection(cleaned_msg)
        if injection_reason:
            reply = RuntimeReply(
                answer=(
                    "I cannot comply with requests to override instructions, reveal system prompts, "
                    "or fabricate unsupported data.\n\nSources: No supporting document found."
                ),
                sources=[],
                confidence="unsupported",
                refused_unsupported_topic=True,
                mode_used="safety_guard",
                error=injection_reason,
            )
            self.memory.add_assistant_message(content=reply.answer, sources=[], tool_calls=[])
            return reply

        try:
            # Resolve follow-up context if message is very short
            retrieval_query = cleaned_msg
            if len(cleaned_msg.split()) <= 5 and self.memory.last_user_topic_context():
                retrieval_query = f"{self.memory.last_user_topic_context()} {cleaned_msg}"

            # 1. Execute retrieval if retriever is configured
            hits: list[RetrievalResult] = []
            if self.retriever is not None:
                try:
                    hits = self.retriever.retrieve(
                        query=retrieval_query,
                        top_k=self.agent_def.retrieval_config.top_k,
                        min_score=self.agent_def.retrieval_config.min_score,
                        hybrid_keyword_weight=self.agent_def.retrieval_config.hybrid_keyword_weight,
                    )
                    tool_traces.append(
                        ToolCallTrace(
                            tool_name="document_retriever",
                            arguments={"query": retrieval_query, "top_k": self.agent_def.retrieval_config.top_k},
                            output_preview=f"Retrieved {len(hits)} chunk(s)",
                            success=True,
                        )
                    )
                except Exception as exc:
                    tool_traces.append(
                        ToolCallTrace(
                            tool_name="document_retriever",
                            arguments={"query": retrieval_query},
                            success=False,
                            error=str(exc),
                        )
                    )

            # 2. Optional auxiliary tool execution (calculator / date_tool / csv_lookup)
            aux_notes = self._run_auxiliary_tools(cleaned_msg, tool_traces)

            retrieved_chunk_ids = [h.chunk.chunk_id for h in hits]
            retrieved_context = [f"[{h.citation}] {h.chunk.content}" for h in hits]
            if aux_notes:
                retrieved_context.extend(aux_notes)

            should_use_llm = (
                has_configured_llm_credentials(self.agent_def.model) if use_llm is None else use_llm
            )

            if should_use_llm:
                try:
                    reply = self._reply_with_llm(
                        user_message=cleaned_msg,
                        hits=hits,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieved_context=retrieved_context,
                        tool_traces=tool_traces,
                    )
                except Exception as llm_exc:
                    reply = self._reply_deterministic(
                        user_message=cleaned_msg,
                        hits=hits,
                        aux_notes=aux_notes,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieved_context=retrieved_context,
                        tool_traces=tool_traces,
                    )
                    reply.mode_used = "deterministic_fallback"
                    reply.error = f"Live LLM unavailable ({llm_exc}); used deterministic grounded fallback."
            else:
                reply = self._reply_deterministic(
                    user_message=cleaned_msg,
                    hits=hits,
                    aux_notes=aux_notes,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_context=retrieved_context,
                    tool_traces=tool_traces,
                )

        except Exception as exc:
            reply = RuntimeReply(
                answer="An internal runtime error occurred while processing your request.",
                confidence="low",
                tool_calls=tool_traces,
                error=str(exc),
            )

        self.memory.add_assistant_message(
            content=reply.answer,
            sources=reply.sources,
            tool_calls=reply.tool_calls,
        )
        return reply

    def _run_auxiliary_tools(self, message: str, traces: list[ToolCallTrace]) -> list[str]:
        notes: list[str] = []

        # Calculator tool for explicit arithmetic expressions
        if "calculator" in self.bound_tools:
            expr_match = re.search(r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)", message)
            if expr_match:
                expr = expr_match.group(1)
                try:
                    res = self.bound_tools["calculator"](expr)
                    traces.append(
                        ToolCallTrace(
                            tool_name="calculator",
                            arguments={"expression": expr},
                            output_preview=str(res),
                            success=True,
                        )
                    )
                    if not str(res).startswith("Error"):
                        notes.append(f"[calculator] {expr} = {res}")
                except Exception as exc:
                    traces.append(ToolCallTrace(tool_name="calculator", arguments={"expression": expr}, success=False, error=str(exc)))

        # CSV lookup tool if enabled
        if "csv_lookup" in self.bound_tools and is_tool_permitted(self.agent_def, "csv_lookup"):
            raw_tokens = re.findall(r"[a-zA-Z0-9_-]{3,}", message)
            stopwords = {"what", "when", "where", "which", "about", "does", "have", "rules", "listed", "find", "part", "number", "price", "from", "with", "that", "this"}
            # Prioritize alphanumeric part IDs (e.g. SP-402) then longer specific nouns
            candidates = sorted(
                [w for w in raw_tokens if w.lower() not in stopwords],
                key=lambda t: (bool(re.search(r"\d", t) and re.search(r"[a-zA-Z]", t)), len(t)),
                reverse=True,
            )
            for tok in candidates[:3]:
                try:
                    raw = self.bound_tools["csv_lookup"](query=tok, limit=4)
                    traces.append(
                        ToolCallTrace(
                            tool_name="csv_lookup",
                            arguments={"query": tok},
                            output_preview=raw[:160],
                            success=True,
                        )
                    )
                    if not raw.startswith("No CSV records matched"):
                        notes.append(f"[csv_lookup] {raw}")
                        break
                except Exception as exc:
                    traces.append(ToolCallTrace(tool_name="csv_lookup", arguments={"query": tok}, success=False, error=str(exc)))

        # Date tool for "today" / current date queries
        if "date_tool" in self.bound_tools and re.search(r"\b(today|current date|what day is it)\b", message, re.I):
            try:
                today_val = self.bound_tools["date_tool"](operation="today")
                traces.append(
                    ToolCallTrace(
                        tool_name="date_tool",
                        arguments={"operation": "today"},
                        output_preview=str(today_val),
                        success=True,
                    )
                )
                notes.append(f"[date_tool] Today's date is {today_val}.")
            except Exception as exc:
                traces.append(ToolCallTrace(tool_name="date_tool", arguments={"operation": "today"}, success=False, error=str(exc)))

        return notes

    def _reply_with_llm(
        self,
        user_message: str,
        hits: list[RetrievalResult],
        retrieved_chunk_ids: list[str],
        retrieved_context: list[str],
        tool_traces: list[ToolCallTrace],
    ) -> RuntimeReply:
        from litellm import completion

        context_block = "\n\n".join(retrieved_context) if retrieved_context else "No retrieved evidence."
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{self.agent_def.system_prompt}\n\n"
                    "Respond strictly as JSON matching schema:\n"
                    f"{json.dumps(_StructuredRuntimeLLMOutput.model_json_schema(), indent=2)}"
                ),
            }
        ]
        # Include prior turns (excluding current user message already at tail)
        prior = self.memory.to_llm_messages()[:-1]
        messages.extend(prior)
        messages.append(
            {
                "role": "user",
                "content": f"## Retrieved Evidence\n{context_block}\n\n## User Question\n{user_message}",
            }
        )

        effective_model = resolve_effective_model(self.agent_def.model)

        try:
            resp = completion(
                model=effective_model,
                messages=messages,
                temperature=self.agent_def.model_config_details.temperature,
                max_tokens=self.agent_def.model_config_details.max_tokens,
                response_format=_StructuredRuntimeLLMOutput,
            )
        except Exception:
            resp = completion(
                model=effective_model,
                messages=messages,
                temperature=self.agent_def.model_config_details.temperature,
                max_tokens=self.agent_def.model_config_details.max_tokens,
            )

        raw_content = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        parsed = _StructuredRuntimeLLMOutput.model_validate_json(match.group(0) if match else raw_content)
        conf = parsed.confidence.lower() if parsed.confidence.lower() in {"high", "medium", "low", "unsupported"} else "medium"
        citations = [h.citation for h in hits if h.chunk.document_name in parsed.sources]

        return RuntimeReply(
            answer=parsed.answer,
            sources=parsed.sources,
            formatted_citations=citations,
            confidence=conf,  # type: ignore[arg-type]
            refused_unsupported_topic=parsed.refused_unsupported_topic,
            mode_used="live_llm",
            tool_calls=tool_traces,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_context=retrieved_context,
        )

    def _reply_deterministic(
        self,
        user_message: str,
        hits: list[RetrievalResult],
        aux_notes: list[str],
        retrieved_chunk_ids: list[str],
        retrieved_context: list[str],
        tool_traces: list[ToolCallTrace],
    ) -> RuntimeReply:
        if self._is_unsupported_query(user_message):
            refusal_msg = (
                f"The uploaded documents do not contain reliable information to answer this request "
                f"for {self.agent_def.agent_name} (unsupported topics: "
                f"{', '.join(self.agent_def.unsupported_topics)}). "
                f"I can help with: {', '.join(self.agent_def.supported_topics)}.\n\n"
                "Sources: No supporting document found."
            )
            return RuntimeReply(
                answer=refusal_msg,
                sources=[],
                formatted_citations=[],
                confidence="unsupported",
                refused_unsupported_topic=True,
                mode_used="deterministic_grounded",
                tool_calls=tool_traces,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_context=retrieved_context,
            )

        if not hits and not aux_notes:
            return RuntimeReply(
                answer="The approved knowledge base does not contain information to answer this question.",
                sources=[],
                formatted_citations=[],
                confidence="low",
                refused_unsupported_topic=False,
                tool_calls=tool_traces,
                retrieved_chunk_ids=[],
                retrieved_context=[],
            )

        q_terms = set(re.findall(r"[a-z0-9]{3,}", user_message.lower()))
        selected_hits = self._select_diverse_hits(hits, q_terms, max_hits=4)
        snippets: list[str] = []
        sources: list[str] = []
        formatted_citations: list[str] = []

        include_full_chunk = "detail_mode: full" in self.agent_def.system_prompt.lower()
        for hit in selected_hits:
            excerpt = (
                hit.chunk.content
                if include_full_chunk
                else self._best_matching_sentences(hit.chunk.content, q_terms)
            )
            doc_name = hit.chunk.document_name
            if doc_name not in sources:
                sources.append(doc_name)
            formatted_citations.append(self._pretty_citation(hit))
            section_note = f" ({hit.chunk.section})" if hit.chunk.section else ""
            snippets.append(f"According to {doc_name}{section_note}: {excerpt}")

        for note in aux_notes:
            snippets.append(note)

        citation_footer = (
            f" Source: {'; '.join(formatted_citations)}" if formatted_citations else ""
        )
        answer = " ".join(snippets) + citation_footer
        confidence = "high" if hits and hits[0].score >= 0.25 else "medium"

        return RuntimeReply(
            answer=answer,
            sources=sources,
            formatted_citations=formatted_citations,
            confidence=confidence,  # type: ignore[arg-type]
            refused_unsupported_topic=False,
            tool_calls=tool_traces,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_context=retrieved_context,
        )

    def _is_unsupported_query(self, question: str) -> bool:
        q_lower = question.lower()
        # Allow policy-escalation questions when user asks how escalation or rules work
        if any(w in q_lower for w in ("escalat", "handbook", "who do i contact", "rule applies to")):
            if "my current" not in q_lower and "calculate my" not in q_lower:
                return False

        personal_cues = (
            "my current monthly salary",
            "my current salary",
            "my monthly salary",
            "my salary",
            "get promoted",
            "my bonus",
            "my net payroll",
            "remaining personal leave",
            "do i have left",
            "my next promotion",
            "my performance rating",
            "my stock option",
            "confidential executive compensation",
            "tax withholding",
        )
        if any(cue in q_lower for cue in personal_cues):
            return True

        for topic in self.agent_def.unsupported_topics:
            tokens = [
                t
                for t in re.findall(r"[a-z0-9]+", topic.lower())
                if t not in {"without", "data", "hris", "calculations", "lookups", "individual", "employee", "records"}
            ]
            if not tokens:
                continue
            if all(re.search(rf"\b{re.escape(tok[:5])}", q_lower) for tok in tokens[:2]):
                return True
            if len(tokens) == 1 and re.search(rf"\b{re.escape(tokens[0][:5])}", q_lower):
                return True
        return False

    @staticmethod
    def _select_diverse_hits(hits: list[RetrievalResult], q_terms: set[str], max_hits: int = 4) -> list[RetrievalResult]:
        if len(hits) <= 1:
            return hits
        chosen: list[RetrievalResult] = [hits[0]]
        seen_docs = {hits[0].chunk.document_name}
        for hit in hits[1:]:
            if hit.chunk.document_name not in seen_docs:
                hay = f"{hit.chunk.section or ''} {hit.chunk.content}".lower()
                if any(t in hay for t in q_terms):
                    chosen.append(hit)
                    seen_docs.add(hit.chunk.document_name)
            if len(chosen) >= max_hits:
                break
        return chosen

    @staticmethod
    def _best_matching_sentences(content: str, q_terms: set[str]) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content) if s.strip()]
        if not sentences:
            return content
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for t in q_terms if t in s.lower()),
            reverse=True,
        )
        return " ".join(scored[:2])

    @staticmethod
    def _pretty_citation(hit: RetrievalResult) -> str:
        c = hit.chunk
        parts = [c.document_name]
        if c.page_number is not None:
            parts.append(f"Page {c.page_number}")
        if c.section:
            parts.append(c.section)
        return " — ".join(parts)

    @staticmethod
    def _load_agent(agent_def: AgentDefinition | str | Path) -> AgentDefinition:
        if isinstance(agent_def, AgentDefinition):
            return agent_def
        return AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))

