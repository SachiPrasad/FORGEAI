"""Concrete, safe implementations for all 6 recommended tools in Feature 5."""

from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import math
import operator
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from forgeai.knowledge.retriever import KnowledgeRetriever

_SAFE_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_MATH_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def make_document_retriever_tool(
    manifest_path: str | Path,
    default_top_k: int = 4,
) -> Callable[[str, int], str]:
    """Bind a KnowledgeRetriever instance from knowledge_manifest.json into a callable tool."""
    retriever = KnowledgeRetriever.from_manifest(manifest_path)

    def document_retriever(query: str, top_k: int = default_top_k) -> str:
        """Search uploaded knowledge documents and return relevant excerpts with citations."""
        results = retriever.retrieve(query=query, top_k=top_k)
        return retriever.format_results(results)

    return document_retriever


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression without using eval().

    Supports +, -, *, /, //, %, **, parentheses, and abs/round/min/max/sqrt/ceil/floor.
    """
    if not expression or not expression.strip():
        return "Error: Empty expression."
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_ast_node(tree.body)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(round(value, 6) if isinstance(value, float) else value)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"


def _eval_ast_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY_OPS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large.")
        return _SAFE_BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _SAFE_MATH_FUNCS or node.keywords:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_eval_ast_node(arg) for arg in node.args]
        return _SAFE_MATH_FUNCS[func_name](*args)
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def csv_lookup(
    csv_paths: list[str | Path],
    query: str = "",
    column: str | None = None,
    exact: bool = False,
    limit: int = 10,
) -> str:
    """Search one or more CSV files for exact or partial record matches."""
    if not csv_paths:
        return "Error: No CSV files configured for lookup."

    q_norm = query.strip().lower()
    matches: list[str] = []

    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for row_idx, row in enumerate(reader, start=1):
                if column:
                    # Case-insensitive column match
                    col_key = next((k for k in row if k.lower() == column.lower()), None)
                    if col_key is None:
                        continue
                    cell_val = (row.get(col_key) or "").strip()
                    hit = (cell_val.lower() == q_norm) if exact else (q_norm in cell_val.lower())
                elif not q_norm:
                    hit = True
                else:
                    row_text = " ".join((v or "") for v in row.values()).lower()
                    hit = any(v.strip().lower() == q_norm for v in row.values()) if exact else (q_norm in row_text)

                if hit:
                    formatted = " | ".join(f"{k}: {(v or '').strip()}" for k, v in row.items())
                    matches.append(f"[{path.name} row {row_idx}] {formatted}")
                    if len(matches) >= limit:
                        return "\n".join(matches)

    return "\n".join(matches) if matches else f"No CSV records matched query='{query}'."


def make_csv_lookup_tool(csv_paths: list[str | Path]) -> Callable[..., str]:
    """Bind specific CSV paths into a clean agent-callable lookup function."""
    resolved = [str(Path(p)) for p in csv_paths]

    def bound_csv_lookup(query: str = "", column: str | None = None, exact: bool = False, limit: int = 10) -> str:
        """Search structured CSV tables and return matching records with row numbers."""
        return csv_lookup(csv_paths=resolved, query=query, column=column, exact=exact, limit=limit)

    bound_csv_lookup.__name__ = "csv_lookup"
    return bound_csv_lookup


def date_tool(
    operation: str = "today",
    date_a: str | None = None,
    date_b: str | None = None,
    days: int = 0,
) -> str:
    """Perform date/time operations: 'today', 'now', 'add_days', 'diff_days', 'business_days', 'weekday'."""
    op = operation.strip().lower()
    if op == "today":
        return dt.date.today().isoformat()
    if op == "now":
        return dt.datetime.now(dt.timezone.utc).isoformat()

    if not date_a:
        return "Error: 'date_a' (YYYY-MM-DD) is required for this operation."

    try:
        parsed_a = dt.date.fromisoformat(date_a.strip())
    except ValueError:
        return f"Error: Invalid ISO date '{date_a}'. Expected YYYY-MM-DD."

    if op == "weekday":
        return parsed_a.strftime("%A")
    if op == "add_days":
        return (parsed_a + dt.timedelta(days=days)).isoformat()

    if not date_b:
        return "Error: 'date_b' (YYYY-MM-DD) is required for date range calculations."
    try:
        parsed_b = dt.date.fromisoformat(date_b.strip())
    except ValueError:
        return f"Error: Invalid ISO date '{date_b}'. Expected YYYY-MM-DD."

    if op == "diff_days":
        return str((parsed_b - parsed_a).days)

    if op == "business_days":
        start, end = (parsed_a, parsed_b) if parsed_a <= parsed_b else (parsed_b, parsed_a)
        sign = 1 if parsed_a <= parsed_b else -1
        count = 0
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                count += 1
            cursor += dt.timedelta(days=1)
        return str(sign * count)

    return f"Error: Unsupported date operation '{operation}'."


def http_api(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    max_chars: int = 8000,
) -> str:
    """Perform a safe HTTP/HTTPS request to retrieve JSON or text from an external API."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"Error: Invalid or unsupported URL '{url}'. Only http/https URLs are allowed."

    req_headers = {"User-Agent": "FORGEAI-Agent/0.1", "Accept": "application/json, text/plain"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url=url.strip(), method=method.upper(), headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if len(body) > max_chars:
                body = body[:max_chars] + "\n...[truncated]..."
            return f"HTTP {resp.status}\n{body}"
    except urllib.error.HTTPError as exc:
        return f"HTTP Error {exc.code}: {exc.reason}"
    except Exception as exc:
        return f"Error calling HTTP API '{url}': {exc}"


def web_search(query: str, max_results: int = 5) -> str:
    """Search external web sources using Tavily (if TAVILY_API_KEY is configured)."""
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: TAVILY_API_KEY environment variable is not configured."

    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query.strip(),
            "max_results": max_results,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return "No web search results found."
        return "\n\n".join(
            f"[{r.get('title', 'Untitled')}]({r.get('url', '')})\n{r.get('content', '')}"
            for r in results[:max_results]
        )
    except Exception as exc:
        return f"Error performing web search: {exc}"

