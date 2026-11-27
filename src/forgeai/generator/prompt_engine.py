"""Prompt generation engine for producing grounded, restriction-aware system prompts."""

from __future__ import annotations

from jinja2 import Template
from pydantic import BaseModel, Field

from forgeai.groundability.schemas import SupportedTopicConfig
from forgeai.intake.llm_client import complete_structured
from forgeai.intake.schemas import AgentRequirements

SYSTEM_PROMPT_TEMPLATE = """You are {{ agent_name }} serving {{ target_audience }} in the {{ domain }} domain.

Purpose:
{{ purpose }}

Your responsibilities:
- Answer questions accurately within your verified scope ({{ supported_topics_str }}).
- Use only the approved knowledge base{% if allowed_sources_str %} ({{ allowed_sources_str }}){% endif %}.
- Cite the source document (and page/section when available) whenever possible.
- Never invent company policies, numbers, or rules, and do not pretend missing documents exist.
- If the information is unavailable in the approved knowledge base, clearly state that the knowledge base does not contain the answer.
- Do not answer unsupported questions as facts.
- Maintain a {{ tone.lower() }} tone and deliver responses as {{ output_format.lower() }}.

{% if partially_supported_topics %}
Partially supported topics (answer cautiously with explicit caveats):
{% for topic in partially_supported_topics %}
- {{ topic }}{% if partial_caveats.get(topic) %}: {{ partial_caveats[topic] }}{% endif %}
{% endfor %}
{% endif %}

{% if unsupported_topics %}
Unsupported topics (MUST be explicitly refused):
{% for topic in unsupported_topics %}
- {{ topic }}
{% endfor %}

Refusal policy:
{{ refusal_policy }}
{% endif %}

{% if restrictions %}
Strict restrictions & guardrails:
{% for rule in restrictions %}
- {{ rule }}
{% endfor %}
{% endif %}

{% if tool_names %}
Available tools:
{% for tool in tools %}
- `{{ tool.name }}`: {{ tool.description }}
{% endfor %}
Always retrieve supporting context before answering policy or factual questions.
{% endif %}

Output contract:
Return responses matching this structure:
- `answer`: Clear response grounded in retrieved evidence (or an explicit refusal if unsupported).
- `sources`: List of cited document names/sections used to form the answer (empty list if refused).
- `confidence`: One of `"high"`, `"medium"`, `"low"`, or `"unsupported"`.
"""


class _PolishedPrompt(BaseModel):
    system_prompt: str = Field(..., min_length=50)


class PromptGenerationEngine:
    """Generate structured, guardrail-enforcing system prompts for target agents."""

    def generate(
        self,
        requirements: AgentRequirements,
        topic_config: SupportedTopicConfig,
        tools: list,
        use_llm: bool = False,
        model: str = "gpt-4o-mini",
    ) -> str:
        rendered = Template(SYSTEM_PROMPT_TEMPLATE).render(
            agent_name=topic_config.agent_name,
            target_audience=topic_config.target_audience,
            domain=topic_config.domain,
            purpose=topic_config.purpose,
            supported_topics_str=", ".join(topic_config.supported_topics) or "approved topics",
            allowed_sources_str=", ".join(topic_config.allowed_sources),
            tone=topic_config.tone,
            output_format=topic_config.output_format,
            partially_supported_topics=topic_config.partially_supported_topics,
            partial_caveats=topic_config.partial_topic_caveats,
            unsupported_topics=topic_config.unsupported_topics,
            refusal_policy=topic_config.refusal_policy,
            restrictions=topic_config.restrictions,
            tools=tools,
            tool_names=[t.name for t in tools],
        )
        cleaned = self._normalize_blank_lines(rendered)

        if use_llm:
            polish_instruction = (
                "Refine the following AI agent system prompt for clarity and precision while preserving EVERY "
                "restriction, refusal policy, supported/unsupported topic list, citation rule, and output contract.\n\n"
                f"{cleaned}"
            )
            polished = complete_structured(
                prompt=polish_instruction,
                response_model=_PolishedPrompt,
                model=model,
                temperature=0.1,
            )
            # Ensure critical safety anchors weren't dropped by LLM polishing
            candidate = polished.system_prompt.strip()
            if self._preserves_safety_anchors(candidate, topic_config):
                return candidate
        return cleaned

    @staticmethod
    def _normalize_blank_lines(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        out: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = line == ""
            if is_blank and prev_blank:
                continue
            out.append(line)
            prev_blank = is_blank
        return "\n".join(out).strip()

    @staticmethod
    def _preserves_safety_anchors(prompt: str, topic_config: SupportedTopicConfig) -> bool:
        lower = prompt.lower()
        if "never invent" not in lower and "do not invent" not in lower:
            return False
        for topic in topic_config.unsupported_topics[:3]:
            if topic.lower() not in lower:
                return False
        return True

