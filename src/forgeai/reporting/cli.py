"""CLI for Feature 9 — Honest Final Report generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from forgeai.reporting.report_generator import FinalReportGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-report",
        description="Feature 9 — Generate honest final_report.md.",
    )
    parser.add_argument("--brief", required=True, help="Brief text or path to brief.txt.")
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--initial-eval", required=True)
    parser.add_argument("--final-eval", required=True)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--out", default="final_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    brief_path = Path(args.brief)
    brief_text = brief_path.read_text(encoding="utf-8") if brief_path.is_file() else args.brief

    FinalReportGenerator().generate(
        brief_text=brief_text,
        requirements=args.requirements,
        manifest=args.manifest,
        agent_def=args.agent,
        eval_dataset=args.evals,
        initial_report=args.initial_eval,
        final_report=args.final_eval,
        optimization_log=args.iterations,
        output_path=args.out,
    )
    Console().print(f"[green]Generated honest final report:[/green] {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

