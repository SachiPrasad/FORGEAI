"""Generated Agent Execution Engine for executing agent.json against evaluation test cases."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai.eval.schemas import AgentResponse
from forgeai.generator.schemas import AgentDefinition
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials
from forgeai.knowledge.retriever import KnowledgeRetriever
from forgeai.knowledge.schemas import RetrievalResult
from forgeai.tools.registry import build_default_registry


class _LLMAgentOutput(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    refused_unsupported_topic: bool = False


class GeneratedAgentRunner:
    """Execute a generated agent.json against questions using its configured tools and prompt."""

    def __init__(self, agent_def: AgentDefinition | str | Path) -> None:
        if isinstance(agent_def, AgentDefinition):
            self.agent_def = agent_def
        else:
            self.agent_def = AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))

        self.registry = build_default_registry()
        self.retriever: KnowledgeRetriever | None = None
        manifest_path = self.agent_def.retrieval_config.knowledge_manifest_path
        if manifest_path and Path(manifest_path).is_file():
            self.retriever = KnowledgeRetriever.from_manifest(manifest_path)

    def run(self, question: str, use_llm: bool | None = None) -> AgentResponse:
        should_use_llm = (
            has_configured_llm_credentials(self.agent_def.model) if use_llm is None else use_llm
        )

        # 1. Retrieve relevant context via configured retrieval_config
        hits: list[RetrievalResult] = []
        if self.retriever is not None:
            hits = self.retriever.retrieve(
                query=question,
                top_k=self.agent_def.retrieval_config.top_k,
                min_score=self.agent_def.retrieval_config.min_score,
                hybrid_keyword_weight=self.agent_def.retrieval_config.hybrid_keyword_weight,
            )

        # Optional CSV lookup if csv_lookup tool is selected on the agent
        csv_hits_text = ""
        csv_tool = next((t for t in self.agent_def.tools if t.name == "csv_lookup" and t.enabled), None)
        if csv_tool and csv_tool.parameters.get("csv_paths"):
            lookup_fn = self.registry.build_callable("csv_lookup", csv_paths=csv_tool.parameters["csv_paths"])
            raw_tokens = re.findall(r"[a-zA-Z0-9_-]{3,}", question)
            stopwords = {"what", "which", "according", "rules", "listed", "find", "part", "number", "price", "from", "with", "that", "this", "together"}
            candidates = sorted(
                [w for w in raw_tokens if w.lower() not in stopwords],
                key=lambda t: (bool(re.search(r"\d", t) and re.search(r"[a-zA-Z]", t)), len(t)),
                reverse=True,
            )
            for tok in candidates[:3]:
                raw_csv = lookup_fn(query=tok, limit=4)
                if not raw_csv.startswith("No CSV records matched"):
                    csv_hits_text = raw_csv
                    break

        retrieved_chunk_ids = [h.chunk.chunk_id for h in hits]
        retrieved_context = [f"[{h.citation}] {h.chunk.content}" for h in hits]
        if csv_hits_text:
            retrieved_context.append(f"[csv_lookup] {csv_hits_text}")

        if should_use_llm:
            return self._run_with_llm(question, hits, retrieved_chunk_ids, retrieved_context)
        return self._run_deterministic(question, hits, retrieved_chunk_ids, retrieved_context)

    def _run_with_llm(
        self,
        question: str,
        hits: list[RetrievalResult],
        retrieved_chunk_ids: list[str],
        retrieved_context: list[str],
    ) -> AgentResponse:
        context_block = "\n\n".join(retrieved_context) if retrieved_context else "No retrieved chunks."
        prompt = (
            f"{self.agent_def.system_prompt}\n\n"
            f"## Retrieved Knowledge Base Context\n{context_block}\n\n"
            f"## User Question\n{question}"
        )
        llm_out = complete_structured(
            prompt=prompt,
            response_model=_LLMAgentOutput,
            model=self.agent_def.model,
            temperature=self.agent_def.model_config_details.temperature,
        )
        conf = llm_out.confidence.lower() if llm_out.confidence.lower() in {"high", "medium", "low", "unsupported"} else "medium"
        raw = {
            "answer": llm_out.answer,
            "sources": llm_out.sources,
            "confidence": conf,
            "refused_unsupported_topic": llm_out.refused_unsupported_topic,
        }
        return AgentResponse(
            answer=llm_out.answer,
            sources=llm_out.sources,
            confidence=conf,  # type: ignore[arg-type]
            refused_unsupported_topic=llm_out.refused_unsupported_topic,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_context=retrieved_context,
            raw_json=raw,
        )

    def _run_deterministic(
        self,
        question: str,
        hits: list[RetrievalResult],
        retrieved_chunk_ids: list[str],
        retrieved_context: list[str],
    ) -> AgentResponse:
        # Check if question hits an unsupported topic or personal/private record scope
        if self._is_unsupported_query(question):
            refusal_msg = (
                f"I cannot answer this request because it falls outside the approved knowledge base "
                f"for {self.agent_def.agent_name}. Unsupported topics include: "
                f"{', '.join(self.agent_def.unsupported_topics)}. "
                f"I can help with: {', '.join(self.agent_def.supported_topics)}."
            )
            raw = {
                "answer": refusal_msg,
                "sources": [],
                "confidence": "unsupported",
                "refused_unsupported_topic": True,
            }
            return AgentResponse(
                answer=refusal_msg,
                sources=[],
                confidence="unsupported",
                refused_unsupported_topic=True,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_context=retrieved_context,
                raw_json=raw,
            )

        if not hits:
            no_info_msg = (
                "The approved knowledge base does not contain information to answer this question."
            )
            raw = {
                "answer": no_info_msg,
                "sources": [],
                "confidence": "low",
                "refused_unsupported_topic": False,
            }
            return AgentResponse(
                answer=no_info_msg,
                sources=[],
                confidence="low",
                refused_unsupported_topic=False,
                retrieved_chunk_ids=[],
                retrieved_context=[],
                raw_json=raw,
            )

        # Pick top grounded sentences across distinct documents/chunks matching query terms
        q_terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        selected_hits = self._select_diverse_hits(hits, q_terms, max_hits=4)
        answer_snippets: list[str] = []
        sources: list[str] = []
        include_full_chunk = "detail_mode: full" in self.agent_def.system_prompt.lower()

        for hit in selected_hits:
            best_sent = (
                hit.chunk.content
                if include_full_chunk
                else self._best_matching_sentence(hit.chunk.content, q_terms)
            )
            doc_name = hit.chunk.document_name
            if doc_name not in sources:
                sources.append(doc_name)
            section_note = f" ({hit.chunk.section})" if hit.chunk.section else ""
            answer_snippets.append(f"According to {doc_name}{section_note}: {best_sent}")

        confidence = "high" if hits[0].score >= 0.25 else "medium"
        answer = " ".join(answer_snippets)
        raw = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "refused_unsupported_topic": False,
        }
        return AgentResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,  # type: ignore[arg-type]
            refused_unsupported_topic=False,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_context=retrieved_context,
            raw_json=raw,
        )

    def _is_unsupported_query(self, question: str) -> bool:
        q_lower = question.lower()
        # Allow policy-escalation questions when user asks how escalation or rules work
        if any(w in q_lower for w in ("escalat", "handbook", "who do i contact", "rule applies to")):
            if "my current" not in q_lower and "calculate my" not in q_lower:
                return False

        # Explicit personal-account / HRIS lookup cues when unsupported
        personal_cues = (
            "my current monthly salary",
            "my current salary",
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
            if all(
                re.search(rf"\b{re.escape(tok[:5])}", q_lower)
                for tok in tokens[:2]
            ):
                return True
        return False

    @staticmethod
    def _select_diverse_hits(hits: list[RetrievalResult], q_terms: set[str], max_hits: int = 3) -> list[RetrievalResult]:
        if len(hits) <= 1:
            return hits
        chosen: list[RetrievalResult] = [hits[0]]
        seen_docs = {hits[0].chunk.document_name}
        # Prioritize chunks from other documents that share query keywords
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
    def _best_matching_sentence(content: str, q_terms: set[str]) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content) if s.strip()]
        if not sentences:
            return content
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for t in q_terms if t in s.lower()),
            reverse=True,
        )
        # Return up to 2 top matching sentences so multi-fact chunks preserve details
        return " ".join(scored[:2])

