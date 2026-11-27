"""Core orchestrator for Feature 4 — Agent Generator -> agent.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forgeai.generator.prompt_engine import PromptGenerationEngine
from forgeai.generator.schemas import AgentDefinition, AgentModelConfig, RetrievalConfig
from forgeai.generator.tool_selector import ToolSelector
from forgeai.generator.validator import AgentDefinitionValidator
from forgeai.groundability.guard import GroundabilityGuard
from forgeai.groundability.schemas import SupportedTopicConfig
from forgeai.intake.llm_client import has_configured_llm_credentials
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.schemas import KnowledgeManifest


class AgentGenerator:
    """Dynamically synthesize a validated agent.json from requirements, groundability config, and KB manifest."""

    def __init__(
        self,
        prompt_engine: PromptGenerationEngine | None = None,
        tool_selector: ToolSelector | None = None,
        validator: AgentDefinitionValidator | None = None,
    ) -> None:
        self.prompt_engine = prompt_engine or PromptGenerationEngine()
        self.tool_selector = tool_selector or ToolSelector()
        self.validator = validator or AgentDefinitionValidator()

    def generate(
        self,
        requirements: AgentRequirements | str | Path,
        supported_topic_config: SupportedTopicConfig | str | Path | None = None,
        knowledge_manifest: KnowledgeManifest | str | Path | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_turns: int = 10,
        top_k: int | None = None,
        chunk_size_override: int | None = None,
        chunk_overlap_override: int | None = None,
        use_llm: bool | None = None,
        output_path: str | Path | None = None,
    ) -> AgentDefinition:
        req = self._load_requirements(requirements)
        manifest_obj, manifest_path_str = self._load_manifest(knowledge_manifest)

        if supported_topic_config is not None:
            topic_cfg = self._load_supported_config(supported_topic_config)
        elif knowledge_manifest is not None and manifest_path_str is not None:
            _, topic_cfg = GroundabilityGuard().evaluate(
                requirements=req,
                manifest=manifest_path_str,
                use_llm=False,
            )
        else:
            topic_cfg = self._fallback_topic_config(req)

        retrieval_config = self._build_retrieval_config(
            manifest=manifest_obj,
            manifest_path_str=manifest_path_str,
            top_k=top_k,
            chunk_size_override=chunk_size_override,
            chunk_overlap_override=chunk_overlap_override,
        )

        tools, rejected_tools, selection_record = self.tool_selector.select_with_record(
            requirements=req,
            topic_config=topic_cfg,
            retrieval_config=retrieval_config,
            manifest=manifest_obj,
        )

        should_use_llm = has_configured_llm_credentials(model) if use_llm is None else use_llm
        system_prompt = self.prompt_engine.generate(
            requirements=req,
            topic_config=topic_cfg,
            tools=tools,
            use_llm=should_use_llm,
            model=model,
        )

        output_contract, output_schema, structured_output_code = self._build_output_spec(topic_cfg)

        model_cfg = AgentModelConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
        )

        agent_def = AgentDefinition(
            agent_name=topic_cfg.agent_name,
            domain=topic_cfg.domain,
            target_audience=topic_cfg.target_audience,
            purpose=topic_cfg.purpose,
            tone=topic_cfg.tone,
            output_format=topic_cfg.output_format,
            model=model,
            model_config_details=model_cfg,
            system_prompt=system_prompt,
            retrieval_config=retrieval_config,
            tools=tools,
            rejected_tools=rejected_tools,
            tool_selection_record=selection_record,
            supported_topics=topic_cfg.supported_topics,
            partially_supported_topics=topic_cfg.partially_supported_topics,
            unsupported_topics=topic_cfg.unsupported_topics,
            restrictions=topic_cfg.restrictions,
            allowed_sources=topic_cfg.allowed_sources,
            output_contract=output_contract,
            output_schema=output_schema,
            structured_output_code=structured_output_code,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generation_mode": "llm_polished" if should_use_llm else "deterministic",
                "groundability_score": topic_cfg.groundability_score,
                "recommendation": topic_cfg.recommendation,
            },
        )

        has_kb = (manifest_obj is not None and manifest_obj.total_chunks > 0) or bool(
            retrieval_config.knowledge_manifest_path
        )
        agent_def.validation = self.validator.validate(agent_def, has_knowledge_base=has_kb)

        if output_path is not None:
            self.save_agent_definition(agent_def, output_path)

        return agent_def

    @staticmethod
    def save_agent_definition(agent_def: AgentDefinition, output_path: str | Path) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = agent_def.model_dump(by_alias=True)
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest

    @staticmethod
    def _build_retrieval_config(
        manifest: KnowledgeManifest | None,
        manifest_path_str: str | None,
        top_k: int | None,
        chunk_size_override: int | None,
        chunk_overlap_override: int | None,
    ) -> RetrievalConfig:
        if manifest is not None:
            chunk_size = chunk_size_override or manifest.chunking.chunk_size
            chunk_overlap = (
                chunk_overlap_override if chunk_overlap_override is not None else manifest.chunking.chunk_overlap
            )
            strategy = manifest.chunking.strategy
            if top_k is not None:
                resolved_top_k = top_k
            elif manifest.total_chunks <= 6:
                resolved_top_k = 4
            elif manifest.total_chunks <= 25:
                resolved_top_k = 5
            else:
                resolved_top_k = 6

            vector_path = manifest.vector_store_path
            if manifest_path_str and not Path(vector_path).is_absolute():
                vector_path = str((Path(manifest_path_str).parent / vector_path).resolve())

            return RetrievalConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=resolved_top_k,
                strategy=strategy,
                knowledge_manifest_path=str(Path(manifest_path_str).resolve()) if manifest_path_str else None,
                vector_store_path=vector_path,
            )

        return RetrievalConfig(
            chunk_size=chunk_size_override or 700,
            chunk_overlap=chunk_overlap_override if chunk_overlap_override is not None else 100,
            top_k=top_k or 4,
        )

    @staticmethod
    def _build_output_spec(topic_cfg: SupportedTopicConfig) -> tuple[dict[str, Any], dict[str, Any], str]:
        contract: dict[str, Any] = {
            "answer": "string",
            "sources": [],
            "confidence": "string",
        }
        if topic_cfg.unsupported_topics:
            contract["refused_unsupported_topic"] = "boolean"

        properties: dict[str, Any] = {
            "answer": {
                "type": "string",
                "description": "Grounded answer to the user's question or explicit refusal if unsupported.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cited source documents, pages, and sections supporting the answer.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "unsupported"],
                "description": "Confidence level based on retrieved document evidence.",
            },
        }
        required = ["answer", "sources", "confidence"]

        if topic_cfg.unsupported_topics:
            properties["refused_unsupported_topic"] = {
                "type": "boolean",
                "description": "True if the request was refused because it fell outside supported topics.",
            }

        schema = {
            "title": "StructuredOutput",
            "type": "object",
            "properties": properties,
            "required": required,
        }

        code_lines = [
            "from typing import Literal",
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class StructuredOutput(BaseModel):",
            '    answer: str = Field(..., description="Grounded answer or explicit refusal if unsupported.")',
            '    sources: list[str] = Field(default_factory=list, description="Cited document sources.")',
            '    confidence: Literal["high", "medium", "low", "unsupported"] = Field(',
            '        ..., description="Evidence-backed confidence level."',
            "    )",
        ]
        if topic_cfg.unsupported_topics:
            code_lines.append(
                '    refused_unsupported_topic: bool = Field(default=False, description="True if refused as unsupported.")'
            )

        return contract, schema, "\n".join(code_lines)

    @staticmethod
    def _load_requirements(requirements: AgentRequirements | str | Path) -> AgentRequirements:
        if isinstance(requirements, AgentRequirements):
            return requirements
        return AgentRequirements.model_validate_json(Path(requirements).read_text(encoding="utf-8"))

    @staticmethod
    def _load_supported_config(cfg: SupportedTopicConfig | str | Path) -> SupportedTopicConfig:
        if isinstance(cfg, SupportedTopicConfig):
            return cfg
        return SupportedTopicConfig.model_validate_json(Path(cfg).read_text(encoding="utf-8"))

    @staticmethod
    def _load_manifest(manifest: KnowledgeManifest | str | Path | None) -> tuple[KnowledgeManifest | None, str | None]:
        if manifest is None:
            return None, None
        if isinstance(manifest, KnowledgeManifest):
            return manifest, None
        path = Path(manifest)
        return KnowledgeManifest.model_validate_json(path.read_text(encoding="utf-8")), str(path)

    @staticmethod
    def _fallback_topic_config(req: AgentRequirements) -> SupportedTopicConfig:
        return SupportedTopicConfig(
            agent_name=req.agent_name,
            domain=req.domain,
            target_audience=req.target_audience,
            purpose=req.purpose,
            tone=req.tone,
            output_format=req.output_format,
            supported_topics=req.supported_topics,
            partially_supported_topics=[],
            unsupported_topics=req.unsupported_topics,
            restrictions=req.restrictions,
            refusal_policy=(
                f"Refuse questions about {', '.join(req.unsupported_topics) or 'unsupported topics'} "
                "and never invent missing information."
            ),
            allowed_sources=req.sources,
            groundability_score=1.0 if req.sources else 0.5,
            recommendation="Generated directly from requirements.json.",
        )

