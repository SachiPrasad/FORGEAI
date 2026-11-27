"""Feature 6 — Automatic Evaluation Dataset Generator (minimum 20 non-hardcoded document-driven cases)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from forgeai.eval.schemas import (
    DEFAULT_CATEGORY_COUNTS,
    MIN_EVAL_CASES,
    EvalCase,
    EvalCaseBatch,
    EvalCategory,
    EvalDataset,
)
from forgeai.generator.schemas import AgentDefinition
from forgeai.intake.llm_client import complete_structured, has_configured_llm_credentials
from forgeai.knowledge.schemas import ChunkMetadata, KnowledgeManifest

_BOUNDARY_PATTERN = re.compile(
    r"\b(?:at\s+least|up\s+to|within|after|before|more\s+than|less\s+than|longer\s+than|"
    r"maximum|minimum|carried\s+over|carry\s+over|eligibility|eligible|must|required|"
    r"\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?|percent|%|usd|\$))\b",
    re.I,
)


@dataclass
class _FactCandidate:
    chunk: ChunkMetadata
    sentence: str
    facts: list[str]
    subject: str
    is_boundary: bool


class EvalDatasetGenerator:
    """Dynamically synthesize >= 20 grounded evaluation test cases from uploaded documents and agent.json."""

    def generate(
        self,
        agent_def: AgentDefinition | str | Path,
        manifest: KnowledgeManifest | str | Path,
        category_counts: dict[EvalCategory, int] | None = None,
        use_llm: bool | None = None,
        model: str = "gpt-4o-mini",
        output_path: str | Path | None = None,
    ) -> EvalDataset:
        agent = self._load_agent(agent_def)
        manifest_obj = self._load_manifest(manifest)

        if not manifest_obj.chunks:
            raise ValueError("KnowledgeManifest has 0 chunks; cannot generate document-grounded evaluation cases.")

        counts = dict(category_counts or DEFAULT_CATEGORY_COUNTS)
        total_requested = sum(counts.values())
        if total_requested < MIN_EVAL_CASES:
            counts["answerable"] = counts.get("answerable", 0) + (MIN_EVAL_CASES - total_requested)

        should_use_llm = has_configured_llm_credentials(model) if use_llm is None else use_llm

        if should_use_llm:
            cases = self._generate_with_llm(agent, manifest_obj, counts, model=model)
            mode = "llm_generated"
        else:
            cases = self._generate_from_documents(agent, manifest_obj, counts)
            mode = "document_synthesized"

        # Deduplicate + top up to guarantee >= MIN_EVAL_CASES unique validated cases
        deduped = self._deduplicate_cases(cases)
        if len(deduped) < MIN_EVAL_CASES:
            extra_pool = self._generate_from_documents(
                agent,
                manifest_obj,
                {
                    "answerable": 15,
                    "multi_document": 6,
                    "edge_case": 5,
                    "ambiguous": 4,
                    "unsupported": 10,
                },
            )
            deduped = self._deduplicate_cases([*deduped, *extra_pool])

        # Re-number cleanly with domain prefix (e.g., HR_001 .. HR_020)
        prefix = self._id_prefix(agent.domain, agent.agent_name)
        final_cases: list[EvalCase] = []
        for idx, case in enumerate(deduped, start=1):
            final_cases.append(case.model_copy(update={"test_id": f"{prefix}_{idx:03d}"}))

        cat_summary: dict[str, int] = {}
        for case in final_cases:
            cat_summary[case.category] = cat_summary.get(case.category, 0) + 1

        dataset = EvalDataset(
            agent_name=agent.agent_name,
            total_cases=len(final_cases),
            category_counts=cat_summary,
            cases=final_cases,
            metadata={
                "generation_mode": mode,
                "model": model if should_use_llm else None,
                "source_documents": [d.document_name for d in manifest_obj.documents],
                "total_chunks_analyzed": manifest_obj.total_chunks,
            },
        )

        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

        return dataset

    def _generate_with_llm(
        self,
        agent: AgentDefinition,
        manifest: KnowledgeManifest,
        counts: dict[EvalCategory, int],
        model: str,
    ) -> list[EvalCase]:
        chunk_excerpts = "\n\n".join(
            f"[{c.chunk_id}] doc={c.document_name} | section={c.section or '-'} | page={c.page_number}\n{c.content[:450]}"
            for c in manifest.chunks[:30]
        )
        total_target = sum(counts.values())
        prompt = (
            f"Generate {total_target} unique, realistic evaluation test cases for '{agent.agent_name}' "
            f"({agent.domain}) based STRICTLY on the document chunks and unsupported topics below.\n\n"
            "Required category distribution:\n"
            + "\n".join(f"- {cat}: {qty}" for cat, qty in counts.items())
            + f"\n\nSupported topics: {agent.supported_topics}\n"
            + f"Unsupported topics (must have should_refuse=true, expected_sources=[]): {agent.unsupported_topics}\n\n"
            + f"Document Chunks:\n{chunk_excerpts}\n\n"
            "Rules:\n"
            "- Use exact document filenames from the chunks for expected_sources.\n"
            "- Extract concrete expected_facts (short phrases or numbers present in the chunks).\n"
            "- Multi-document cases must require combining facts from >= 2 different documents.\n"
            "- No duplicate or near-duplicate questions."
        )
        batch = complete_structured(prompt=prompt, response_model=EvalCaseBatch, model=model, temperature=0.3)
        return batch.cases

    def _generate_from_documents(
        self,
        agent: AgentDefinition,
        manifest: KnowledgeManifest,
        counts: dict[EvalCategory, int],
    ) -> list[EvalCase]:
        candidates = self._mine_fact_candidates(manifest.chunks)
        cases: list[EvalCase] = []

        # 1. Direct answerable questions (default 8)
        cases.extend(self._build_answerable_cases(candidates, target=counts.get("answerable", 8)))

        # 2. Multi-document questions (default 4)
        cases.extend(self._build_multi_doc_cases(candidates, manifest.chunks, target=counts.get("multi_document", 4)))

        # 3. Edge-case questions (default 3)
        cases.extend(self._build_edge_cases(candidates, agent, target=counts.get("edge_case", 3)))

        # 4. Ambiguous questions (default 2)
        cases.extend(self._build_ambiguous_cases(candidates, agent, target=counts.get("ambiguous", 2)))

        # 5. Unsupported/out-of-scope questions (default 3)
        cases.extend(self._build_unsupported_cases(agent, target=counts.get("unsupported", 3)))

        return cases

    def _mine_fact_candidates(self, chunks: list[ChunkMetadata]) -> list[_FactCandidate]:
        mined: list[_FactCandidate] = []
        for chunk in chunks:
            if chunk.document_type == "csv":
                for line in chunk.content.splitlines():
                    if line.startswith("Row "):
                        facts = self._extract_facts_from_text(line)
                        subject = self._extract_csv_subject(line) or (chunk.section or chunk.document_name)
                        mined.append(
                            _FactCandidate(
                                chunk=chunk,
                                sentence=line,
                                facts=facts,
                                subject=subject,
                                is_boundary=bool(_BOUNDARY_PATTERN.search(line)),
                            )
                        )
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", chunk.content) if len(s.strip()) >= 20]
            for sent in sentences:
                facts = self._extract_facts_from_text(sent)
                subject = self._extract_sentence_subject(sent, chunk.section)
                mined.append(
                    _FactCandidate(
                        chunk=chunk,
                        sentence=sent,
                        facts=facts,
                        subject=subject,
                        is_boundary=bool(_BOUNDARY_PATTERN.search(sent)),
                    )
                )
        return mined

    def _build_answerable_cases(self, candidates: list[_FactCandidate], target: int) -> list[EvalCase]:
        cases: list[EvalCase] = []
        if not candidates:
            return cases

        # Spread across chunks/documents first so all documents get coverage
        ordered = self._interleave_by_document(candidates)
        for idx in range(target):
            cand = ordered[idx % len(ordered)]
            variant = idx // len(ordered)
            section_label = cand.chunk.section or cand.chunk.document_name
            question = self._phrase_direct_question(cand, section_label, variant=variant)
            cases.append(
                EvalCase(
                    test_id=f"ANS_{idx + 1}",
                    category="answerable",
                    question=question,
                    expected_answer=cand.sentence,
                    expected_sources=[cand.chunk.document_name],
                    expected_facts=cand.facts[:4],
                    should_refuse=False,
                    source_chunk_ids=[cand.chunk.chunk_id],
                )
            )
        return cases

    def _build_multi_doc_cases(
        self,
        candidates: list[_FactCandidate],
        chunks: list[ChunkMetadata],
        target: int,
    ) -> list[EvalCase]:
        cases: list[EvalCase] = []
        by_doc: dict[str, list[_FactCandidate]] = {}
        for c in candidates:
            by_doc.setdefault(c.chunk.document_name, []).append(c)

        doc_names = list(by_doc.keys())
        if len(doc_names) < 2:
            # Fallback if only 1 document exists: pair two distinct sections/chunks
            if len(candidates) >= 2:
                for i in range(target):
                    a = candidates[i % len(candidates)]
                    b = candidates[(i + 1) % len(candidates)]
                    cases.append(
                        EvalCase(
                            test_id=f"MULTI_{i + 1}",
                            category="multi_document",
                            question=(
                                f"Compare and summarize the requirements for '{a.subject}' and '{b.subject}' "
                                f"across the documented sections ({a.chunk.section or a.chunk.document_name} "
                                f"and {b.chunk.section or b.chunk.document_name})."
                            ),
                            expected_answer=f"{a.sentence} {b.sentence}",
                            expected_sources=self._dedupe([a.chunk.document_name, b.chunk.document_name]),
                            expected_facts=self._dedupe([*a.facts[:2], *b.facts[:2]]),
                            should_refuse=False,
                            source_chunk_ids=self._dedupe([a.chunk.chunk_id, b.chunk.chunk_id]),
                        )
                    )
            return cases

        pairs: list[tuple[_FactCandidate, _FactCandidate]] = []
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                c1 = by_doc[doc_names[i]][0]
                c2 = by_doc[doc_names[j]][0]
                pairs.append((c1, c2))

        for idx in range(target):
            c1, c2 = pairs[idx % len(pairs)]
            # Pick varied candidates within those two docs on later passes
            list1 = by_doc[c1.chunk.document_name]
            list2 = by_doc[c2.chunk.document_name]
            c1 = list1[idx % len(list1)]
            c2 = list2[(idx + 1) % len(list2)]
            sec1 = c1.chunk.section or c1.chunk.document_name
            sec2 = c2.chunk.section or c2.chunk.document_name
            question = (
                f"Considering both {sec1} and {sec2}, what rules apply to {c1.subject} "
                f"together with {c2.subject}?"
            )
            cases.append(
                EvalCase(
                    test_id=f"MULTI_{idx + 1}",
                    category="multi_document",
                    question=question,
                    expected_answer=f"{c1.sentence} {c2.sentence}",
                    expected_sources=[c1.chunk.document_name, c2.chunk.document_name],
                    expected_facts=self._dedupe([*c1.facts[:2], *c2.facts[:2]]),
                    should_refuse=False,
                    source_chunk_ids=[c1.chunk.chunk_id, c2.chunk.chunk_id],
                )
            )
        return cases

    def _build_edge_cases(
        self,
        candidates: list[_FactCandidate],
        agent: AgentDefinition,
        target: int,
    ) -> list[EvalCase]:
        cases: list[EvalCase] = []
        boundary_cands = [c for c in candidates if c.is_boundary] or candidates
        if not boundary_cands:
            return cases

        for idx in range(target):
            cand = boundary_cands[idx % len(boundary_cands)]
            sec = cand.chunk.section or cand.chunk.document_name
            if idx == 0 and agent.partially_supported_topics:
                partial_topic = agent.partially_supported_topics[0]
                # Find any chunk mentioning partial topic
                partial_cand = next(
                    (c for c in candidates if any(w in c.sentence.lower() for w in partial_topic.lower().split())),
                    cand,
                )
                cases.append(
                    EvalCase(
                        test_id=f"EDGE_{idx + 1}",
                        category="edge_case",
                        question=(
                            f"What exact coverage limits or exceptions exist for {partial_topic}, "
                            "and what details are missing from the documents?"
                        ),
                        expected_answer=partial_cand.sentence,
                        expected_sources=[partial_cand.chunk.document_name],
                        expected_facts=partial_cand.facts[:3],
                        should_refuse=False,
                        source_chunk_ids=[partial_cand.chunk.chunk_id],
                    )
                )
                continue

            question = (
                f"What specific threshold, deadline, or eligibility condition applies under {sec} "
                f"for {cand.subject}?"
            )
            cases.append(
                EvalCase(
                    test_id=f"EDGE_{idx + 1}",
                    category="edge_case",
                    question=question,
                    expected_answer=cand.sentence,
                    expected_sources=[cand.chunk.document_name],
                    expected_facts=cand.facts[:4],
                    should_refuse=False,
                    source_chunk_ids=[cand.chunk.chunk_id],
                )
            )
        return cases

    def _build_ambiguous_cases(
        self,
        candidates: list[_FactCandidate],
        agent: AgentDefinition,
        target: int,
    ) -> list[EvalCase]:
        cases: list[EvalCase] = []
        topics = agent.supported_topics or ["policy"]
        for idx in range(target):
            topic = topics[idx % len(topics)]
            matching = [
                c
                for c in candidates
                if topic.lower() in c.sentence.lower() or topic.lower() in (c.chunk.section or "").lower()
            ]
            chosen = matching[:2] if matching else candidates[:2]
            expected_sources = self._dedupe([c.chunk.document_name for c in chosen])
            expected_facts = self._dedupe([f for c in chosen for f in c.facts[:2]])[:4]
            expected_answer = " ".join(c.sentence for c in chosen)
            prompts = [
                f"How does {topic.lower()} work for my situation?",
                f"What rules or limits apply to {topic.lower()} if I have an unusual case?",
                f"Am I eligible under the {topic.lower()} guidelines?",
            ]
            cases.append(
                EvalCase(
                    test_id=f"AMB_{idx + 1}",
                    category="ambiguous",
                    question=prompts[idx % len(prompts)],
                    expected_answer=expected_answer,
                    expected_sources=expected_sources,
                    expected_facts=expected_facts,
                    should_refuse=False,
                    source_chunk_ids=[c.chunk.chunk_id for c in chosen],
                )
            )
        return cases

    def _build_unsupported_cases(self, agent: AgentDefinition, target: int) -> list[EvalCase]:
        cases: list[EvalCase] = []
        unsupported = list(agent.unsupported_topics)
        fallback_out_of_scope = [
            "Confidential executive compensation records",
            "Personal tax withholding calculations",
            "Private employee performance ratings",
            "Stock option vesting schedule for an individual",
            "Live bank account routing numbers",
            "Personal home address records",
            "Confidential board meeting minutes",
            "Private medical claim reimbursement history",
            "Individual disciplinary action records",
            "Real-time ERP financial ledger entries",
        ]
        pool = unsupported + [x for x in fallback_out_of_scope if x not in unsupported]

        for idx in range(target):
            topic = pool[idx % len(pool)]
            topic_clean = topic.rstrip(".")
            if "salary" in topic_clean.lower():
                question = "What is my current monthly salary and bonus payout?"
            elif "promotion" in topic_clean.lower():
                question = "When am I eligible for my next promotion and grade increase?"
            elif "payroll" in topic_clean.lower():
                question = "Can you calculate my net payroll after tax deductions for this month?"
            elif "leave balance" in topic_clean.lower():
                question = "How many remaining personal leave days do I have left in my account right now?"
            elif "erp" in topic_clean.lower() or "purchase order" in topic_clean.lower():
                question = "Can you execute a purchase order in the live ERP system right now?"
            elif "customs" in topic_clean.lower():
                question = "What is the live port customs clearance status for shipment #8841?"
            else:
                question = f"Can you look up private or real-time records for {topic_clean} (case {idx + 1})?"

            cases.append(
                EvalCase(
                    test_id=f"UNS_{idx + 1}",
                    category="unsupported",
                    question=question,
                    expected_answer=f"Refuse because '{topic_clean}' is unsupported by available documents.",
                    expected_sources=[],
                    expected_facts=[],
                    should_refuse=True,
                )
            )
        return cases

    @staticmethod
    def _phrase_direct_question(cand: _FactCandidate, section_label: str, variant: int = 0) -> str:
        if cand.chunk.document_type == "csv":
            csv_templates = [
                f"According to {cand.chunk.document_name}, what rule or details are listed for '{cand.subject}'?",
                f"In {cand.chunk.document_name}, what is specified regarding {cand.subject}?",
                f"Can you retrieve the catalog or table entry for {cand.subject} from {cand.chunk.document_name}?",
                f"What values or limits appear in {cand.chunk.document_name} for {cand.subject}?",
            ]
            return csv_templates[variant % len(csv_templates)]
        templates = [
            f"According to the {section_label} section, what policy applies to {cand.subject}?",
            f"What does {cand.chunk.document_name} state about {cand.subject} under {section_label}?",
            f"Please explain the rule for {cand.subject} documented in {section_label}.",
            f"What specific figures or requirements are defined for {cand.subject}?",
            f"Could you summarize the documented guidance on {cand.subject}?",
            f"Under {section_label}, how is {cand.subject} handled?",
            f"Where is {cand.subject} covered and what exact details are provided?",
            f"Verify what the official {cand.chunk.document_name} specifies for {cand.subject}.",
        ]
        return templates[variant % len(templates)]

    @staticmethod
    def _extract_csv_subject(line: str) -> str:
        # e.g. "Row 1: rule_id: ATT-01 | category: Core Hours | rule_summary: ..."
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and ":" in parts[1]:
            return parts[1].split(":", 1)[1].strip()
        if parts and ":" in parts[0]:
            return parts[0].split(":", 1)[-1].strip()
        return "record"

    @staticmethod
    def _extract_sentence_subject(sentence: str, section: str | None) -> str:
        # Take meaningful leading noun phrase or section fallback
        cleaned = re.sub(r"^(?:the|all|a|an|full-time|unused|new)\s+", "", sentence.strip(), flags=re.I)
        words = re.findall(r"[a-zA-Z0-9-]+", cleaned)[:5]
        if len(words) >= 2:
            return " ".join(words).lower()
        return (section or "policy rules").lower()

    @staticmethod
    def _extract_facts_from_text(text: str) -> list[str]:
        facts: list[str] = []
        # Extract quantitative terms first (numbers + units)
        for match in re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:days?|business days?|calendar days?|weeks?|months?|years?|hours?|minutes?|percent|%|USD|\$)?\b",
            text,
            flags=re.I,
        ):
            tok = match.strip()
            if tok and tok not in facts and len(tok) >= 2:
                facts.append(tok)
        # Extract key domain words (>= 5 chars)
        stopwords = {
            "employees",
            "employee",
            "company",
            "policy",
            "receive",
            "within",
            "following",
            "after",
            "before",
            "about",
            "through",
            "between",
            "remaining",
            "required",
            "must",
            "should",
        }
        for word in re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()):
            if word not in stopwords and word not in facts:
                facts.append(word)
            if len(facts) >= 5:
                break
        return facts[:5]

    @staticmethod
    def _interleave_by_document(candidates: list[_FactCandidate]) -> list[_FactCandidate]:
        buckets: dict[str, list[_FactCandidate]] = {}
        for c in candidates:
            buckets.setdefault(c.chunk.document_name, []).append(c)
        ordered: list[_FactCandidate] = []
        idx = 0
        while True:
            added = False
            for doc_list in buckets.values():
                if idx < len(doc_list):
                    ordered.append(doc_list[idx])
                    added = True
            if not added:
                break
            idx += 1
        return ordered

    @staticmethod
    def _deduplicate_cases(cases: list[EvalCase]) -> list[EvalCase]:
        unique: list[EvalCase] = []
        seen_norms: list[set[str]] = []
        seen_exact: set[str] = set()

        for case in cases:
            q_clean = re.sub(r"[^a-z0-9\s]", "", case.question.lower()).strip()
            q_clean = re.sub(r"\s+", " ", q_clean)
            if not q_clean or q_clean in seen_exact:
                continue
            tokens = set(q_clean.split())
            is_near_dup = False
            for prev_tokens in seen_norms:
                union = tokens | prev_tokens
                if union and (len(tokens & prev_tokens) / len(union)) >= 0.88:
                    is_near_dup = True
                    break
            if is_near_dup:
                continue
            seen_exact.add(q_clean)
            seen_norms.append(tokens)
            unique.append(case)
        return unique

    @staticmethod
    def _id_prefix(domain: str, agent_name: str) -> str:
        if "human resources" in domain.lower() or "hr" in agent_name.lower():
            return "HR"
        letters = re.findall(r"[A-Za-z]+", agent_name.upper())
        if letters:
            return letters[0][:4]
        return "EVAL"

    @staticmethod
    def _load_agent(agent_def: AgentDefinition | str | Path) -> AgentDefinition:
        if isinstance(agent_def, AgentDefinition):
            return agent_def
        return AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))

    @staticmethod
    def _load_manifest(manifest: KnowledgeManifest | str | Path) -> KnowledgeManifest:
        if isinstance(manifest, KnowledgeManifest):
            return manifest
        return KnowledgeManifest.model_validate_json(Path(manifest).read_text(encoding="utf-8"))

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

