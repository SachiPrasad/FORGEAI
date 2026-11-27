"""CLI for Feature 4 — Agent Generator -> agent.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.generator.generator import AgentGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-generate",
        description="Feature 4 — Agent Generator: synthesize validated agent.json blueprint.",
    )
    parser.add_argument("--requirements", required=True, help="Path to requirements.json from Feature 1.")
    parser.add_argument(
        "--groundability",
        default=None,
        help="Path to supported_topic_config.json from Feature 3.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to knowledge_manifest.json from Feature 2.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Configured model ID for the generated agent.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic prompt synthesis without LLM call.")
    parser.add_argument("--out", default="agent.json", help="Output path for agent.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()

    agent_def = AgentGenerator().generate(
        requirements=args.requirements,
        supported_topic_config=args.groundability,
        knowledge_manifest=args.manifest,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        use_llm=False if args.no_llm else None,
        output_path=args.out,
    )

    table = Table(title=f"Generated Agent Configuration — {agent_def.agent_name}", show_lines=True)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Configuration")

    table.add_row("Model", f"{agent_def.model} (temp={agent_def.model_config_details.temperature}, max_turns={agent_def.model_config_details.max_turns})")
    table.add_row(
        "Retrieval Config",
        f"strategy={agent_def.retrieval_config.strategy}, chunk_size={agent_def.retrieval_config.chunk_size}, "
        f"overlap={agent_def.retrieval_config.chunk_overlap}, top_k={agent_def.retrieval_config.top_k}",
    )
    table.add_row("Tools Selected", ", ".join(t.name for t in agent_def.tools) or "None")
    table.add_row("Supported Topics", ", ".join(agent_def.supported_topics) or "None")
    table.add_row("Partial Topics", ", ".join(agent_def.partially_supported_topics) or "None")
    table.add_row("Unsupported Topics", ", ".join(agent_def.unsupported_topics) or "None")
    table.add_row("Output Contract", ", ".join(f"{k}: {type(v).__name__ if not isinstance(v, str) else v}" for k, v in agent_def.output_contract.items()))
    table.add_row(
        "Validation",
        f"valid={agent_def.validation.is_valid}, score={agent_def.validation.completeness_score:.0%}, "
        f"issues={len(agent_def.validation.issues)}",
    )
    console.print(table)
    console.print(Panel(agent_def.system_prompt, title="Generated System Prompt", border_style="green"))
    console.print(f"[green]Saved agent definition to:[/green] {Path(args.out).resolve()}")
    return 0 if agent_def.validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

