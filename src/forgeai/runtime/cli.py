"""Interactive & single-turn CLI for Feature 10 — Generated Agent Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from forgeai.runtime.runner import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-chat",
        description="Feature 10 — Chat with a generated FORGEAI agent (agent.json).",
    )
    parser.add_argument("--agent", required=True, help="Path to generated agent.json.")
    parser.add_argument("--message", "-m", default=None, help="Single question to ask (non-interactive).")
    parser.add_argument("--history-file", default=None, help="Optional JSON path to persist chat history.")
    parser.add_argument("--no-llm", action="store_true", help="Run in offline deterministic grounded mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()

    runtime = AgentRuntime.from_file(args.agent)
    if args.history_file:
        runtime.memory.load(args.history_file)

    use_llm = False if args.no_llm else None

    if args.message:
        reply = runtime.chat(args.message, use_llm=use_llm)
        console.print(
            Panel(
                reply.answer,
                title=f"{runtime.agent_def.agent_name} (mode={reply.mode_used}, confidence={reply.confidence})",
                border_style="yellow" if reply.refused_unsupported_topic else "green",
            )
        )
        if args.history_file:
            runtime.memory.save(args.history_file)
        return 0

    console.print(
        Panel(
            f"Loaded [bold]{runtime.agent_def.agent_name}[/bold] ({runtime.agent_def.domain})\n"
            f"Supported: {', '.join(runtime.agent_def.supported_topics)}\n"
            "Commands: /history, /clear, /exit",
            title="FORGEAI Agent Runtime",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if user_input.lower() == "/clear":
            runtime.memory.clear()
            console.print("[dim]Conversation memory cleared.[/dim]")
            continue
        if user_input.lower() == "/history":
            for turn in runtime.memory.get_history():
                console.print(f"[bold]{turn.role}:[/bold] {turn.content}")
            continue

        reply = runtime.chat(user_input, use_llm=use_llm)
        console.print(
            Panel(
                reply.answer,
                title=f"{runtime.agent_def.agent_name} (mode={reply.mode_used}, confidence={reply.confidence})",
                border_style="yellow" if reply.refused_unsupported_topic else "green",
            )
        )

    if args.history_file:
        saved = runtime.memory.save(args.history_file)
        console.print(f"[green]Saved chat history to:[/green] {Path(saved).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

