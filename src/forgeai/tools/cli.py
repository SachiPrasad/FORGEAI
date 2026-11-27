"""CLI for Feature 5 — Tool Registry & Dynamic Tool Selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from forgeai.tools.registry import build_default_registry
from forgeai.tools.selector import DynamicToolSelector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-tools",
        description="Feature 5 — Tool Registry & Dynamic Tool Selection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List all tools in the centralized registry.")
    list_cmd.add_argument("--export", default=None, help="Optional path to export tool_registry.json.")

    sel_cmd = sub.add_parser("select", help="Dynamically select tools for an agent with explicit reasons.")
    sel_cmd.add_argument("--requirements", required=True, help="Path to requirements.json.")
    sel_cmd.add_argument("--groundability", default=None, help="Optional path to supported_topic_config.json.")
    sel_cmd.add_argument("--manifest", default=None, help="Optional path to knowledge_manifest.json.")
    sel_cmd.add_argument("--top-k", type=int, default=4)
    sel_cmd.add_argument("--no-llm", action="store_true")
    sel_cmd.add_argument("--out", required=True, help="Output directory for tool_registry.json & tool_selection.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()
    registry = build_default_registry()

    if args.command == "list":
        table = Table(title="FORGEAI Centralized Tool Registry", show_lines=True)
        table.add_column("Tool ID", style="cyan")
        table.add_column("Display Name", style="white")
        table.add_column("Priority", style="magenta")
        table.add_column("Purpose")
        for entry in registry.list_entries():
            table.add_row(entry.tool_id, entry.display_name, entry.priority, entry.purpose)
        console.print(table)
        if args.export:
            out_path = registry.save_json(args.export)
            console.print(f"[green]Exported registry to:[/green] {out_path.resolve()}")
        return 0

    if args.command == "select":
        selector = DynamicToolSelector(registry=registry)
        record = selector.select(
            requirements=args.requirements,
            topic_config=args.groundability,
            manifest=args.manifest,
            top_k=args.top_k,
            use_llm=False if args.no_llm else None,
            output_dir=args.out,
        )
        table = Table(title=f"Tool Selection Record — {record.agent_name}", show_lines=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Priority", style="magenta")
        table.add_column("Selected")
        table.add_column("Reason")
        for d in record.decisions:
            sel_str = "[green]Yes[/green]" if d.selected else "[dim]No[/dim]"
            table.add_row(d.display_name, d.priority, sel_str, d.reason)
        console.print(table)
        console.print(f"[bold]Selected tools:[/bold] {record.selected_tools}")
        console.print(f"[green]Saved outputs to:[/green] {Path(args.out).resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

