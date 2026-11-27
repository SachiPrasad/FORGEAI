"""Validator for generated AgentDefinition (agent.json)."""

from __future__ import annotations

from forgeai.generator.schemas import (
    AgentDefinition,
    AgentValidationIssue,
    AgentValidationReport,
)


class AgentDefinitionValidator:
    """Validate prompt safety, retrieval parameters, tools, topic isolation, and output contract."""

    def validate(
        self,
        agent_def: AgentDefinition,
        has_knowledge_base: bool = True,
    ) -> AgentValidationReport:
        issues: list[AgentValidationIssue] = []
        checks_passed: list[str] = []

        # 1. Core identity & model
        if not agent_def.agent_name.strip():
            issues.append(AgentValidationIssue(severity="error", component="agent_name", message="agent_name is empty."))
        else:
            checks_passed.append("agent_name_present")

        if not agent_def.model.strip():
            issues.append(AgentValidationIssue(severity="error", component="model", message="model is empty."))
        else:
            checks_passed.append("model_configured")

        # 2. System prompt & safety checks
        prompt_lower = agent_def.system_prompt.lower()
        if len(agent_def.system_prompt.strip()) < 40:
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="system_prompt",
                    message="system_prompt is too short or missing responsibilities.",
                )
            )
        else:
            checks_passed.append("system_prompt_non_empty")

        if not any(kw in prompt_lower for kw in ("never invent", "do not invent", "knowledge base", "approved")):
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="system_prompt",
                    message="system_prompt must include explicit anti-hallucination / knowledge-base grounding rules.",
                )
            )
        else:
            checks_passed.append("anti_hallucination_rules_present")

        if agent_def.unsupported_topics:
            if "unsupported" not in prompt_lower and "refus" not in prompt_lower:
                issues.append(
                    AgentValidationIssue(
                        severity="error",
                        component="system_prompt",
                        message="unsupported_topics are defined, but system_prompt lacks refusal instructions.",
                    )
                )
            else:
                checks_passed.append("unsupported_refusal_instructions_present")

        # 3. Topic isolation
        overlap = {t.lower() for t in agent_def.supported_topics} & {t.lower() for t in agent_def.unsupported_topics}
        if overlap:
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="topics",
                    message=f"Topics appear in both supported and unsupported lists: {sorted(overlap)}.",
                )
            )
        else:
            checks_passed.append("supported_unsupported_disjoint")

        if not agent_def.supported_topics and not agent_def.partially_supported_topics:
            issues.append(
                AgentValidationIssue(
                    severity="warning",
                    component="topics",
                    message="No supported or partially supported topics configured.",
                )
            )

        # 4. Retrieval & tools consistency
        tool_names = {t.name for t in agent_def.tools if t.enabled}
        if has_knowledge_base and not ({"document_retriever", "retrieve_knowledge"} & tool_names):
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="tools",
                    message="Knowledge base is present, but a retrieval tool ('document_retriever') is missing.",
                )
            )
        elif has_knowledge_base:
            checks_passed.append("retrieval_tool_attached")

        restrictions_lower = " ".join(agent_def.restrictions).lower()
        if "strictly" in restrictions_lower and ({"web_search", "search_tavily"} & tool_names):
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="tools",
                    message="External web search tool selected despite closed-book KB restrictions.",
                )
            )
        else:
            checks_passed.append("tool_policy_consistent")

        # 5. Output contract verification
        required_contract_keys = {"answer", "sources", "confidence"}
        missing_keys = required_contract_keys - set(agent_def.output_contract.keys())
        if missing_keys:
            issues.append(
                AgentValidationIssue(
                    severity="error",
                    component="output_contract",
                    message=f"output_contract missing required keys: {sorted(missing_keys)}.",
                )
            )
        else:
            checks_passed.append("output_contract_valid")

        has_errors = any(i.severity == "error" for i in issues)
        total_checks = max(1, len(checks_passed) + len([i for i in issues if i.severity == "error"]))
        score = round(len(checks_passed) / total_checks, 2)

        return AgentValidationReport(
            is_valid=not has_errors,
            completeness_score=score,
            checks_passed=checks_passed,
            issues=issues,
        )

