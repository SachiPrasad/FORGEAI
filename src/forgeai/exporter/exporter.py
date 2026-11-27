"""Export a generated FORGEAI agent with prompts, config, tools, dependencies, and runner script."""

from __future__ import annotations

import shutil
from pathlib import Path

from forgeai.generator.generator import AgentGenerator
from forgeai.generator.schemas import AgentDefinition

_RUNNER_SCRIPT_TEMPLATE = '''"""Standalone runnable entrypoint for {agent_name}."""

from __future__ import annotations

import argparse
from pathlib import Path

from forgeai.runtime import AgentRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Run {agent_name}")
    parser.add_argument("--message", "-m", default=None, help="Single question to ask (omit for interactive chat).")
    parser.add_argument("--use-llm", action="store_true", help="Force LLM mode using configured API credentials.")
    args = parser.parse_args()

    agent_path = Path(__file__).resolve().parent / "agent.json"
    runtime = AgentRuntime.from_file(agent_path)
    use_llm = True if args.use_llm else False

    if args.message:
        reply = runtime.chat(args.message, use_llm=use_llm)
        print(reply.answer)
        return 0

    print(f"Loaded {{runtime.agent_def.agent_name}}. Type 'exit' to quit.")
    while True:
        user_msg = input("You> ").strip()
        if user_msg.lower() in {{"exit", "quit"}}:
            break
        reply = runtime.chat(user_msg, use_llm=use_llm)
        print(f"\\nAgent> {{reply.answer}}\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


class AgentExporter:
    """Package agent.json, self-contained knowledge base copy, runner script, requirements, and README."""

    def export(
        self,
        agent_def: AgentDefinition | str | Path,
        output_dir: str | Path,
        copy_knowledge_base: bool = True,
    ) -> dict[str, Path]:
        agent = (
            agent_def
            if isinstance(agent_def, AgentDefinition)
            else AgentDefinition.model_validate_json(Path(agent_def).read_text(encoding="utf-8"))
        )

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        exported_agent = agent.model_copy(deep=True)

        # Copy knowledge base artifacts into bundle/kb for portability
        if copy_knowledge_base and agent.retrieval_config.knowledge_manifest_path:
            src_manifest = Path(agent.retrieval_config.knowledge_manifest_path)
            if src_manifest.is_file():
                kb_dir = out_root / "kb"
                kb_dir.mkdir(parents=True, exist_ok=True)
                dest_manifest = kb_dir / src_manifest.name
                shutil.copy2(src_manifest, dest_manifest)

                if agent.retrieval_config.vector_store_path:
                    src_db = Path(agent.retrieval_config.vector_store_path)
                    if src_db.is_file():
                        dest_db = kb_dir / src_db.name
                        shutil.copy2(src_db, dest_db)
                        exported_agent.retrieval_config.vector_store_path = str(dest_db.resolve())

                exported_agent.retrieval_config.knowledge_manifest_path = str(dest_manifest.resolve())
                for tool in exported_agent.tools:
                    if tool.name == "document_retriever":
                        tool.parameters["manifest_path"] = str(dest_manifest.resolve())

        agent_path = AgentGenerator.save_agent_definition(exported_agent, out_root / "agent.json")
        prompt_path = out_root / "system_prompt.txt"
        prompt_path.write_text(exported_agent.system_prompt, encoding="utf-8")

        runner_path = out_root / "run_agent.py"
        runner_path.write_text(_RUNNER_SCRIPT_TEMPLATE.format(agent_name=exported_agent.agent_name), encoding="utf-8")

        req_path = out_root / "requirements.txt"
        req_path.write_text(
            "\n".join(
                [
                    "jinja2>=3.1.0",
                    "litellm>=1.0.0",
                    "numpy>=1.26.0",
                    "pydantic>=2.0.0",
                    "pypdf>=5.0.0",
                    "python-docx>=1.1.0",
                    "python-dotenv>=1.0.0",
                    "rich>=13.0.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        readme_path = out_root / "README.md"
        readme_path.write_text(
            f"# {exported_agent.agent_name}\n\n"
            f"**Domain:** {exported_agent.domain}\n\n"
            f"**Purpose:** {exported_agent.purpose}\n\n"
            f"## Tools Enabled\n"
            + "\n".join(f"- `{t.name}`: {t.description}" for t in exported_agent.tools)
            + "\n\n## Run\n```bash\npython run_agent.py -m \"Ask your question here\"\n```\n",
            encoding="utf-8",
        )

        return {
            "agent_json": agent_path,
            "system_prompt": prompt_path,
            "run_agent": runner_path,
            "requirements": req_path,
            "readme": readme_path,
        }

