"""CLI for Features 6 & 7 — Evaluation Dataset Generator & Evaluation Judge."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.eval.dataset_generator import EvalDatasetGenerator
from forgeai.eval.judge import EvaluationJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-eval",
        description="Features 6 & 7 — Automatic Eval Dataset Generator & Hybrid Evaluation Judge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate >= 20 document-driven test cases into evals.json.")
    gen.add_argument("--agent", required=True, help="Path to agent.json from Feature 4.")
    gen.add_argument("--manifest", required=True, help="Path to knowledge_manifest.json from Feature 2.")
    gen.add_argument("--out", default="evals.json", help="Output path for evals.json.")
    gen.add_argument("--model", default="gpt-4o-mini")
    gen.add_argument("--no-llm", action="store_true")

    judge = sub.add_parser("judge", help="Execute generated agent on evals.json and produce scorecard.")
    judge.add_argument("--agent", required=True, help="Path to agent.json from Feature 4.")
    judge.add_argument("--evals", required=True, help="Path to evals.json from Feature 6.")
    judge.add_argument("--out", required=True, help="Output directory for eval_results.json & scorecard.md.")
    judge.add_argument("--model", default="gpt-4o-mini")
    judge.add_argument("--no-llm", action="store_true")

    run_all = sub.add_parser("run-all", help="Generate evals.json AND execute + judge the agent end-to-end.")
    run_all.add_argument("--agent", required=True, help="Path to agent.json from Feature 4.")
    run_all.add_argument("--manifest", required=True, help="Path to knowledge_manifest.json from Feature 2.")
    run_all.add_argument("--out", required=True, help="Output directory for evals.json, eval_results.json, scorecard.md.")
    run_all.add_argument("--model", default="gpt-4o-mini")
    run_all.add_argument("--no-llm", action="store_true")
    return parser


def _print_report(console: Console, report) -> None:
    table = Table(title=f"Evaluation Results — {report.scorecard.agent_name}", show_lines=True)
    table.add_column("Test", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Grounded")
    table.add_column("Correct")
    table.add_column("Refusal")
    table.add_column("Sources")
    table.add_column("Score")

    for r in report.results:
        score_styled = "[green]Pass[/green]" if r.passed else "[red]Fail[/red]"
        table.add_row(r.test_id, r.category, r.grounded, r.correct, r.refusal, r.source_quality, score_styled)

    console.print(table)
    sc = report.scorecard
    console.print(
        Panel(
            f"[bold]Pass Rate:[/bold] {sc.passed_cases}/{sc.total_cases} ({sc.pass_rate:.0%})\n"
            f"[bold]Groundedness:[/bold] {sc.groundedness_rate:.0%} | "
            f"[bold]Correctness:[/bold] {sc.correctness_rate:.0%} | "
            f"[bold]Refusal Accuracy:[/bold] {sc.refusal_accuracy_rate:.0%} | "
            f"[bold]Source Quality:[/bold] {sc.source_quality_rate:.0%}",
            title="Evaluation Scorecard",
            border_style="green" if sc.pass_rate >= 0.8 else "yellow",
        )
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()
    use_llm = False if args.no_llm else None

    if args.command == "generate":
        dataset = EvalDatasetGenerator().generate(
            agent_def=args.agent,
            manifest=args.manifest,
            use_llm=use_llm,
            model=args.model,
            output_path=args.out,
        )
        console.print(f"[green]Generated {dataset.total_cases} evaluation cases:[/green] {dataset.category_counts}")
        console.print(f"Saved: {Path(args.out).resolve()}")
        return 0

    if args.command == "judge":
        report = EvaluationJudge().evaluate(
            agent_def=args.agent,
            eval_dataset=args.evals,
            use_llm=use_llm,
            judge_model=args.model,
            output_dir=args.out,
        )
        _print_report(console, report)
        return 0

    if args.command == "run-all":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        evals_path = out_dir / "evals.json"
        dataset = EvalDatasetGenerator().generate(
            agent_def=args.agent,
            manifest=args.manifest,
            use_llm=use_llm,
            model=args.model,
            output_path=evals_path,
        )
        console.print(f"[green]Generated {dataset.total_cases} cases -> {evals_path.resolve()}[/green]")
        report = EvaluationJudge().evaluate(
            agent_def=args.agent,
            eval_dataset=dataset,
            use_llm=use_llm,
            judge_model=args.model,
            output_dir=out_dir,
        )
        _print_report(console, report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

