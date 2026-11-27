"""CLI for Feature 8 — Critic & Optimizer Loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from forgeai.optimizer.optimizer import CriticOptimizerLoop
from forgeai.optimizer.schemas import MAX_OPTIMIZATION_ITERATIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-optimize",
        description="Feature 8 — Critic & Optimizer Loop (max 3 iterations) -> iteration_log.json",
    )
    parser.add_argument("--agent", required=True, help="Path to initial agent.json.")
    parser.add_argument("--evals", required=True, help="Path to evals.json.")
    parser.add_argument("--out", required=True, help="Output directory for iteration_log.json & final eval results.")
    parser.add_argument("--agent-out", default=None, help="Path to write optimized agent.json (defaults to --agent).")
    parser.add_argument("--max-iterations", type=int, default=MAX_OPTIMIZATION_ITERATIONS)
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()
    agent_out = args.agent_out or args.agent

    _, log, _, final_report = CriticOptimizerLoop().optimize(
        agent_def=args.agent,
        eval_dataset=args.evals,
        max_iterations=args.max_iterations,
        use_llm=False if args.no_llm else None,
        output_dir=args.out,
        save_optimized_agent_path=agent_out,
    )

    table = Table(title=f"Critic & Optimizer Loop — {log.agent_name}", show_lines=True)
    table.add_column("Iteration", justify="right", style="cyan")
    table.add_column("Failure Type", style="magenta")
    table.add_column("Change Applied")
    table.add_column("Score Before -> After")
    table.add_column("Rolled Back")

    for rec in log.iterations:
        table.add_row(
            str(rec.iteration),
            rec.failure_type,
            rec.change_applied,
            f"{rec.score_before:.0%} -> {rec.score_after:.0%}",
            "Yes" if rec.rolled_back else "No",
        )
    console.print(table)
    console.print(
        f"[bold]Initial:[/bold] {log.initial_passed}/{log.total_cases} ({log.initial_score:.0%}) | "
        f"[bold green]Final:[/bold green] {log.final_passed}/{log.total_cases} ({log.final_score:.0%})\n"
        f"[dim]Stop reason: {log.stop_reason}[/dim]\n"
        f"[green]Saved iteration_log.json to:[/green] {(Path(args.out) / 'iteration_log.json').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

