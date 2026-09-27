"""
Typer CLI main entry point for all use cases.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from .config import settings
from .logging_setup import setup_logging

app = typer.Typer(
    name="ci",
    help="Customer Intelligence - AI powered customer interaction analysis and insight generation",
    add_completion=False,
)

console = Console(highlight=False, force_terminal=True, color_system="truecolor")


@app.callback()
def _global_options(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging (prints prompts, responses, and latency)",
        is_eager=True,
    ),
    log_file: str = typer.Option(
        "",
        "--log-file",
        help="Also write logs to this file (always at DEBUG level)",
    ),
) -> None:
    """Global options for the CLI."""
    setup_logging(level=logging.DEBUG if debug else logging.INFO, log_file=log_file, force=True)


@app.command()
def analyze(
    text: Optional[str] = typer.Option(None, "--text", help="Raw conversation text to analyze"),
    file: Optional[Path] = typer.Option(None, "--file", help="Path to a text file containing the conversation"),
    index: Optional[int] = typer.Option(None, "--index", help="Row index (0-based) from the processed dataset"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON instead of a formatted panel"),
) -> None:
    """Analyze a single customer conversation (Use Case 1)."""
    from .interaction_analysis import analyze_interaction

    if text:
        conversation_text = text
    elif file:
        conversation_text = file.read_text(encoding="utf-8")
    elif index is not None:
        from .data_loader import get_or_build_processed
        conversations = get_or_build_processed()
        conversation_text = conversations[index].raw_text
    else:
        rprint("[red]Provide one of --text, --file, or --index[/red]")
        raise typer.Exit(code=1)

    result = analyze_interaction(conversation_text)

    if as_json:
        print(result.model_dump_json(indent=2))
        return

    body = (
        f"[bold]Summary:[/bold] {result.formal_summary}\n\n"
        f"[bold]Topic:[/bold] {result.primary_topic} (confidence {result.topic_confidence:.2f})\n"
        f"[bold]Intent:[/bold] {result.customer_intent}\n"
        f"[bold]Sentiment:[/bold] {result.sentiment}\n"
        f"[bold]Temperature:[/bold] {result.temperature_band} ({result.temperature_score}/100)\n"
        f"[bold]Escalation required:[/bold] {result.escalation_required}"
        + (f" - {result.escalation_reason}" if result.escalation_required else "")
        + f"\n[bold]Next action:[/bold] {result.recommended_next_action}\n\n"
        f"[dim]Evidence: {result.evidence}[/dim]"
    )
    if result.ungrounded_evidence:
        body += f"\n[red]Ungrounded evidence flagged: {result.ungrounded_evidence}[/red]"
    console.print(Panel(body, title="Interaction Analysis", expand=False))


@app.command("batch-analyze")
def batch_analyze(
    n: Optional[int] = typer.Option(None, "--n", help="Number of rows to analyze (default: all)"),
    out: Path = typer.Option(Path("data/processed/analyzed.jsonl"), "--out", help="Output JSONL path"),
) -> None:
    """Run interaction analysis over a batch of conversations from the raw dataset (Use Case 1)."""
    from .data_loader import get_or_build_processed
    from .interaction_analysis import analyze_batch, save_analyzed

    conversations = get_or_build_processed()
    if n is not None:
        conversations = conversations[:n]

    with console.status(f"Analyzing {len(conversations)} conversations..."):
        records = analyze_batch(conversations)

    save_analyzed(records, out)
    console.print(f"[green]Analyzed {len(records)}/{len(conversations)} conversations -> {out}[/green]")


@app.command()
def themes(
    analyzed_path: Path = typer.Option(Path("data/processed/analyzed.jsonl"), "--analyzed", help="Path to analyzed.jsonl (run `batch-analyze` first if missing)"),
) -> None:
    """Identify recurring themes and detect emerging issues (Use Case 3)."""
    from .interaction_analysis import load_analyzed
    from .data_loader import load_synthetic
    from .theme_analysis import build_theme_report

    if not analyzed_path.exists():
        rprint(f"[red]{analyzed_path} not found - run `ci batch-analyze` first.[/red]")
        raise typer.Exit(code=1)

    records = load_analyzed(analyzed_path)
    emerging_candidates = load_synthetic()

    with console.status("Clustering themes and checking for emerging issues..."):
        report = build_theme_report(records, emerging_conversations=emerging_candidates)

    table = Table(title="Customer Themes")
    table.add_column("Theme")
    table.add_column("Count", justify="right")
    table.add_column("Avg Temp", justify="right")
    table.add_column("Temperature bands")
    for theme in report.themes:
        bands = ", ".join(f"{k}:{v}" for k, v in theme.temperature_band_distribution.items())
        table.add_row(theme.label, str(theme.count), f"{theme.avg_temperature_score:.1f}", bands)
    console.print(table)

    if report.emerging_issues:
        for issue in report.emerging_issues:
            console.print(Panel(
                f"{issue.summary}\n\n[dim]Conversations: {issue.example_conversation_ids}[/dim]",
                title=f"[yellow]Emerging Issue: {issue.label} (x{issue.count})[/yellow]",
            ))
    else:
        console.print("[dim]No emerging issues detected.[/dim]")


@app.command()
def ingest(
    kb_dir: Path = typer.Option(Path(settings.kb_dir), "--kb-dir", help="Directory of knowledge base markdown files"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Directory to persist the FAISS index"),
) -> None:
    """Build the RAG knowledge base index (Use Case 4)."""
    from .rag.ingest import build_index

    resolved_out = out_dir or (Path(settings.artifacts_dir) / "rag")
    with console.status(f"Ingesting knowledge base from {kb_dir}..."):
        store, chunks = build_index(kb_dir, resolved_out)
    console.print(f"[green]Indexed {len(chunks)} chunks ({store.size} vectors) -> {resolved_out}[/green]")


@app.command()
def ask(question: str) -> None:
    """Ask a question against the RAG knowledge base (Use Case 4)."""
    from .rag.qa import answer as rag_answer

    result = rag_answer(question)
    style = "green" if result.answered else "yellow"
    console.print(Panel(result.answer, title=f"[{style}]Answer[/{style}]"))
    if result.citations:
        table = Table(title="Citations")
        table.add_column("#")
        table.add_column("Title")
        table.add_column("Source")
        table.add_column("Score", justify="right")
        for i, cit in enumerate(result.citations, 1):
            table.add_row(str(i), cit.title, cit.source, f"{cit.score:.3f}")
        console.print(table)


@app.command("eval")
def run_eval_command(
    out: Path = typer.Option(Path("artifacts/eval/eval_report.json"), "--out", help="Where to write the JSON eval report"),
) -> None:
    """Run the evaluation suite covering grounding, consistency, and RAG correctness (Use Case 5)."""
    from .evaluation import run_evaluation

    with console.status("Running evaluation suite..."):
        report = run_evaluation()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = Table(title="Evaluation Summary")
    table.add_column("Check")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    for check in report["checks"]:
        table.add_row(check["name"], str(check["passed"]), str(check["total"]))
    console.print(table)
    console.print(f"[dim]Full report written to {out}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()