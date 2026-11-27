"""CLI for Feature 2 — Document Ingestion & Knowledge Processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.knowledge.pipeline import KnowledgeProcessor
from forgeai.knowledge.retriever import KnowledgeRetriever
from forgeai.knowledge.schemas import ChunkingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgeai-knowledge",
        description="Feature 2 — Document Ingestion & Knowledge Processing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a document folder into knowledge_manifest.json + vector DB.")
    ingest.add_argument("--docs", required=True, help="Path to folder containing PDF, DOCX, TXT, CSV, MD files.")
    ingest.add_argument("--out", required=True, help="Output directory for knowledge_manifest.json & vector DB.")
    ingest.add_argument(
        "--strategy",
        choices=["recursive", "section", "fixed", "csv_rows"],
        default="recursive",
        help="Chunking strategy (default: recursive).",
    )
    ingest.add_argument("--chunk-size", type=int, default=600, help="Target max characters per chunk.")
    ingest.add_argument("--chunk-overlap", type=int, default=100, help="Character overlap between chunks.")
    ingest.add_argument(
        "--provider",
        choices=["auto", "local", "litellm"],
        default="auto",
        help="Embedding provider (default: auto).",
    )
    ingest.add_argument("--model", default=None, help="LiteLLM embedding model name (optional).")

    query = sub.add_parser("query", help="Run semantic retrieval against a built knowledge base.")
    query.add_argument("--manifest", required=True, help="Path to knowledge_manifest.json.")
    query.add_argument("--q", required=True, help="Natural language search query.")
    query.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    query.add_argument("--doc-type", choices=["pdf", "docx", "txt", "csv", "md"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    console = Console()

    if args.command == "ingest":
        processor = KnowledgeProcessor()
        manifest = processor.process(
            docs_dir=args.docs,
            output_dir=args.out,
            chunking=ChunkingConfig(
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ),
            embedding_provider=args.provider,
            embedding_model=args.model,
        )

        table = Table(title="Knowledge Base Ingestion Summary", show_lines=True)
        table.add_column("Document", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Chunks", justify="right")
        table.add_column("Sections")
        for doc in manifest.documents:
            table.add_row(
                doc.document_name,
                doc.document_type,
                str(doc.chunk_count),
                ", ".join(doc.sections_detected[:4]) or "-",
            )
        console.print(table)
        console.print(
            f"[green]Processed {manifest.total_documents} documents -> {manifest.total_chunks} chunks[/green]\n"
            f"Manifest: {(Path(args.out) / 'knowledge_manifest.json').resolve()}\n"
            f"Vector DB: {(Path(args.out) / manifest.vector_store_path).resolve()}"
        )
        return 0

    if args.command == "query":
        retriever = KnowledgeRetriever.from_manifest(args.manifest)
        results = retriever.retrieve(query=args.q, top_k=args.top_k, document_type=args.doc_type)
        if not results:
            console.print("[yellow]No matching chunks found.[/yellow]")
            return 0
        for idx, res in enumerate(results, start=1):
            console.print(
                Panel(
                    res.chunk.content,
                    title=f"#{idx} | {res.citation} | chunk_id={res.chunk.chunk_id} | score={res.score:.3f}",
                    border_style="blue",
                )
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

