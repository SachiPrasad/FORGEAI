"""CLI for Feature 1 — Intelligent Intake Interviewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.intake.interviewer import IntakeInterviewer
from forgeai.intake.schemas import AgentRequirements


def _render_summary(console: Console, req: AgentRequirements) -> None:
    table = Table(title=f"Extracted Requirements: {req.agent_name}", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Agent name", req.agent_name)
    table.add_row("Domain", req.domain)
    table.add_row("Target audience", req.target_audience)
    table.add_row("Purpose", req.purpose)
    table.add_row("Supported topics", ", ".join(req.supported_topics))
    table.add_row("Unsupported topics", ", ".join(req.unsupported_topics) or "None")
    table.add_row("Tone", req.tone)
    table.add_row("Output format", req.output_format)
    table.add_row("Restrictions", "; ".join(req.restrictions))
    table.add_row("Sources", ", ".join(req.sources) or "None")
    table.add_row(
        "Validation",
        f"valid={req.validation.is_valid}, completeness={req.validation.completeness_score:.0%}, "
        f"issues={len(req.validation.issues)}",
    )
    console.print(table)

    if req.clarification_questions:
        lines = [
            f"{idx}. ({q.id}) {q.question}" + (f"\n   Answer: {q.answer}" if q.answer else "")
            for idx, q in enumerate(req.clarification_questions, start=1)
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title=f"Clarification Questions ({len(req.clarification_questions)}/5 max)",
                border_style="yellow",
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-intake",
        description="Feature 1 — Intelligent Intake Interviewer -> requirements.json",
    )
    parser.add_argument("--brief", required=True, help="Business brief text or path to a brief .txt file.")
    parser.add_argument("--docs", required=True, help="Path to folder containing PDF, DOCX, TXT, CSV documents.")
    parser.add_argument("--config", default=None, help="Optional path to user config JSON file.")
    parser.add_argument("--output", default="requirements.json", help="Output path for requirements.json.")
    parser.add_argument(
        "--answers",
        default=None,
        help="Optional path to JSON file mapping question IDs (e.g. q1) to stakeholder answers.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt interactively for clarification answers and refine requirements.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()
    interviewer = IntakeInterviewer()

    answers_map: dict[str, str] | None = None
    if args.answers:
        answers_map = json.loads(Path(args.answers).read_text(encoding="utf-8"))

    requirements = interviewer.run(
        brief=args.brief,
        docs_dir=args.docs,
        config=args.config,
        output_path=None if args.interactive else args.output,
        clarification_answers=answers_map,
    )
    _render_summary(console, requirements)

    if args.interactive and requirements.clarification_questions:
        collected: dict[str, str] = {}
        console.print("\n[bold yellow]Please answer the clarification questions (press Enter to skip):[/bold yellow]")
        for q in requirements.clarification_questions:
            if q.suggested_options:
                console.print(f"[dim]Options: {' | '.join(q.suggested_options)}[/dim]")
            reply = console.input(f"[bold]{q.id}[/bold] {q.question}\n> ").strip()
            if reply:
                collected[q.id] = reply
        if collected:
            requirements = interviewer.refine_with_answers(
                requirements=requirements,
                answers=collected,
                user_config=args.config,
                output_path=args.output,
            )
            console.print("\n[bold green]Updated requirements after clarification:[/bold green]")
            _render_summary(console, requirements)
        else:
            interviewer.save_requirements(requirements, args.output)
    elif args.interactive:
        interviewer.save_requirements(requirements, args.output)

    console.print(f"\n[green]Saved requirements to:[/green] {Path(args.output).resolve()}")
    return 0 if requirements.validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
