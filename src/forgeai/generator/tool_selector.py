"""Groundability-aware tool selection delegating to the centralized Feature 5 ToolRegistry."""

from __future__ import annotations

import re

from forgeai.generator.schemas import RetrievalConfig, ToolSpec
from forgeai.groundability.schemas import SupportedTopicConfig
from forgeai.intake.schemas import AgentRequirements
from forgeai.knowledge.schemas import KnowledgeManifest
from forgeai.tools.registry import ToolRegistry, build_default_registry
from forgeai.tools.schemas import ToolSelectionRecord
from forgeai.tools.selector import DynamicToolSelector


class ToolSelector:
    """Select the minimal, safe set of tools from the centralized registry."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()
        self.dynamic_selector = DynamicToolSelector(registry=self.registry)

    def select(
        self,
        requirements: AgentRequirements,
        topic_config: SupportedTopicConfig,
        retrieval_config: RetrievalConfig,
        manifest: KnowledgeManifest | None = None,
    ) -> tuple[list[ToolSpec], list[dict[str, str]]]:
        tools, rejected, _ = self.select_with_record(
            requirements=requirements,
            topic_config=topic_config,
            retrieval_config=retrieval_config,
            manifest=manifest,
        )
        return tools, rejected

    def select_with_record(
        self,
        requirements: AgentRequirements,
        topic_config: SupportedTopicConfig,
        retrieval_config: RetrievalConfig,
        manifest: KnowledgeManifest | None = None,
        use_llm: bool | None = False,
    ) -> tuple[list[ToolSpec], list[dict[str, str]], ToolSelectionRecord]:
        record = self.dynamic_selector.select(
            requirements=requirements,
            topic_config=topic_config,
            manifest=manifest,
            manifest_path=retrieval_config.knowledge_manifest_path,
            top_k=retrieval_config.top_k,
            use_llm=use_llm,
        )

        selected: list[ToolSpec] = []
        rejected: list[dict[str, str]] = []

        for decision in record.decisions:
            entry = self.registry.get_entry(decision.tool_id)
            if decision.selected:
                selected.append(
                    ToolSpec(
                        name=decision.tool_id,
                        type=entry.category,
                        description=entry.description,
                        selection_reason=decision.reason,
                        parameters=decision.parameters,
                    )
                )
            else:
                rejected.append({"name": decision.tool_id, "reason": decision.reason})

        # Add optional guardrail / NLP utility tools when needed
        if topic_config.unsupported_topics or topic_config.partially_supported_topics:
            selected.append(
                ToolSpec(
                    name="verify_topic_scope",
                    type="guardrail",
                    description=(
                        "Check whether a user query falls under supported, partially supported, or "
                        "unsupported topics before generating a response."
                    ),
                    selection_reason=(
                        "Unsupported or partially supported topics were detected by Groundability Guard."
                    ),
                    parameters={
                        "supported_topics": topic_config.supported_topics,
                        "partially_supported_topics": topic_config.partially_supported_topics,
                        "unsupported_topics": topic_config.unsupported_topics,
                    },
                )
            )

        purpose_text = f"{requirements.purpose} {topic_config.purpose}".lower()
        if re.search(r"\b(summariz\w*|digest|briefing|condense)\b", purpose_text):
            selected.append(
                ToolSpec(
                    name="summarize_text_with_llm",
                    type="nlp_utility",
                    description="Summarize retrieved long policy passages into concise user-friendly points.",
                    selection_reason="Agent purpose explicitly requests summarization.",
                    parameters={"summary_length": "concise bullet points"},
                )
            )

        if re.search(r"\b(translat\w*|multilingual|spanish|french|german|hindi)\b", purpose_text):
            selected.append(
                ToolSpec(
                    name="translate_text_with_llm",
                    type="nlp_utility",
                    description="Translate verified policy text into the user's requested language.",
                    selection_reason="Agent purpose explicitly requests translation or multilingual support.",
                    parameters={},
                )
            )

        return selected, rejected, record
