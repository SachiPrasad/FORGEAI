"""Centralized Tool Registry for FORGEAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from forgeai.tools.implementations import (
    calculator,
    csv_lookup,
    date_tool,
    http_api,
    make_csv_lookup_tool,
    make_document_retriever_tool,
    web_search,
)
from forgeai.tools.schemas import ToolRegistryEntry, ToolRegistryExport


class ToolRegistry:
    """Centralized registry holding tool definitions and runtime callable factories."""

    def __init__(self) -> None:
        self._entries: dict[str, ToolRegistryEntry] = {}
        self._factories: dict[str, Callable[..., Callable[..., Any]]] = {}

    def register(
        self,
        entry: ToolRegistryEntry,
        callable_factory: Callable[..., Callable[..., Any]],
    ) -> None:
        self._entries[entry.tool_id] = entry
        self._factories[entry.tool_id] = callable_factory

    def get_entry(self, tool_id: str) -> ToolRegistryEntry:
        if tool_id not in self._entries:
            raise KeyError(f"Tool '{tool_id}' is not registered.")
        return self._entries[tool_id]

    def list_entries(self) -> list[ToolRegistryEntry]:
        return list(self._entries.values())

    def has_tool(self, tool_id: str) -> bool:
        return tool_id in self._entries

    def build_callable(self, tool_id: str, **kwargs: Any) -> Callable[..., Any]:
        if tool_id not in self._factories:
            raise KeyError(f"Tool '{tool_id}' is not registered.")
        return self._factories[tool_id](**kwargs)

    def export(self) -> ToolRegistryExport:
        entries = self.list_entries()
        return ToolRegistryExport(total_tools=len(entries), tools=entries)

    def save_json(self, output_path: str | Path) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.export().model_dump_json(indent=2), encoding="utf-8")
        return dest


def build_default_registry() -> ToolRegistry:
    """Construct the standard FORGEAI centralized registry with all 6 core tools."""
    registry = ToolRegistry()

    registry.register(
        ToolRegistryEntry(
            tool_id="document_retriever",
            display_name="Document Retriever",
            category="knowledge_retrieval",
            priority="essential",
            purpose="Search uploaded knowledge",
            description="Search ingested documents in the vector knowledge base and return cited excerpts.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
            requires_knowledge_base=True,
            safe_for_closed_book=True,
        ),
        lambda manifest_path, top_k=4, **_: make_document_retriever_tool(
            manifest_path=manifest_path, default_top_k=top_k
        ),
    )

    registry.register(
        ToolRegistryEntry(
            tool_id="calculator",
            display_name="Calculator",
            category="computation",
            priority="essential",
            purpose="Arithmetic and calculations",
            description="Safely evaluate arithmetic and mathematical expressions (totals, percentages, averages).",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g. '20 * 12 / 5'."}
                },
                "required": ["expression"],
            },
            safe_for_closed_book=True,
        ),
        lambda **_: calculator,
    )

    registry.register(
        ToolRegistryEntry(
            tool_id="csv_lookup",
            display_name="CSV Lookup",
            category="structured_data",
            priority="essential",
            purpose="Search tables, product catalogs",
            description="Search structured CSV files for exact or filtered row-level record lookups.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "column": {"type": "string"},
                    "exact": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 10},
                },
            },
            requires_csv_files=True,
            safe_for_closed_book=True,
        ),
        lambda csv_paths=None, **_: make_csv_lookup_tool(csv_paths or []) if csv_paths else csv_lookup,
    )

    registry.register(
        ToolRegistryEntry(
            tool_id="date_tool",
            display_name="Date & Time",
            category="temporal",
            priority="recommended",
            purpose="Current date, time, date calculations",
            description="Get current date/time and compute day differences, business days, or date offsets.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["today", "now", "add_days", "diff_days", "business_days", "weekday"],
                    },
                    "date_a": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "date_b": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "days": {"type": "integer", "default": 0},
                },
            },
            safe_for_closed_book=True,
        ),
        lambda **_: date_tool,
    )

    registry.register(
        ToolRegistryEntry(
            tool_id="http_api",
            display_name="HTTP API",
            category="external_api",
            priority="optional",
            purpose="Retrieve information from APIs",
            description="Fetch structured JSON or text responses from external HTTP/HTTPS REST APIs.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "GET"},
                },
                "required": ["url"],
            },
            requires_network=True,
            safe_for_closed_book=False,
        ),
        lambda **_: http_api,
    )

    registry.register(
        ToolRegistryEntry(
            tool_id="web_search",
            display_name="Web Search",
            category="web_search",
            priority="optional",
            purpose="Search external sources",
            description="Search public web sources for external or up-to-date information.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            requires_network=True,
            safe_for_closed_book=False,
        ),
        lambda **_: web_search,
    )

    return registry

