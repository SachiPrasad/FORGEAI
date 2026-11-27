"""CLI for Feature 3 — Groundability Guard."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.groundability.guard import GroundabilityGuard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-groundability",
        description="Feature 3 — Groundability Guard: verify topic coverage against ingested documents.",
    )
    parser.add_argument("--requirements", required=True, help="Path to requirements.json from Feature 1.")
    parser.add_argument("--manifest", required=True, help="Path to knowledge_manifest.json from Feature 2.")
    parser.add_argument("--out", required=True, help="Output directory for groundability report & supported config.")
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Optional extra requested topics to evaluate alongside requirements.json.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top chunks retrieved per topic.")
    parser.add_argument(
        "--strict-no-partial",
        action="store_true",
        help="Exclude partially supported topics from supported_topic_config.json.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model when LLM auditing is enabled.")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic semantic/grounding verification.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()

    guard = GroundabilityGuard()
    report, config = guard.evaluate(
        requirements=args.requirements,
        manifest=args.manifest,
        extra_topics=args.topics,
        top_k=args.top_k,
        allow_partial_topics=not args.strict_no_partial,
        use_llm=False if args.no_llm else None,
        model=args.model,
        output_dir=args.out,
    )

    table = Table(title=f"GROUNDABILITY REPORT — {report.agent_name}", show_lines=True)
    table.add_column("Requested topic", style="cyan")
    table.add_column("Document support", style="magenta")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Supporting documents")

    status_styles = {
        "supported": "[green]Supported[/green]",
        "partial": "[yellow]Partial[/yellow]",
        "unsupported": "[red]Unsupported[/red]",
    }
    for a in report.topic_assessments:
        table.add_row(
            a.topic,
            a.document_support,
            status_styles[a.status],
            f"{a.semantic_score:.2f}",
            ", ".join(a.supporting_documents) or "-",
        )
    console.print(table)
    console.print(
        Panel(
            f"[bold]Groundability score:[/bold] {report.groundability_score:.0%}\n"
            f"[bold]Recommendation:[/bold] {report.recommendation}\n"
            f"[bold]Refusal policy:[/bold] {config.refusal_policy}",
            title="Guard Summary",
            border_style="green" if report.groundability_score >= 0.5 else "red",
        )
    )
    out_dir = Path(args.out).resolve()
    console.print(
        f"[green]Saved:[/green]\n"
        f"- {out_dir / 'groundability_report.md'}\n"
        f"- {out_dir / 'groundability_report.json'}\n"
        f"- {out_dir / 'supported_topic_config.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

