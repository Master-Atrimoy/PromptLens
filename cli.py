"""
CLI entrypoint for Prompt Diff Visualizer.
Usage: python cli.py --v1 prompts/v1.txt --v2 prompts/v2.txt --models llama3 mistral
"""

from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from core.config import get_config
from core.ollama_client import OllamaClient
from core.pipeline import run_diff
from core.schemas import ShiftLevel

app = typer.Typer(help="Prompt Diff Visualizer — semantic diff for LLM prompts")
console = Console()

LEVEL_STYLES = {
    ShiftLevel.TRIVIAL:     "green",
    ShiftLevel.MINOR:       "cyan",
    ShiftLevel.MODERATE:    "yellow",
    ShiftLevel.MAJOR:       "orange3",
    ShiftLevel.FUNDAMENTAL: "red",
}


@app.command()
def diff(
    v1: str = typer.Option(..., "--v1", help="Path to prompt v1 file or inline text"),
    v2: str = typer.Option(..., "--v2", help="Path to prompt v2 file or inline text"),
    models: list[str] = typer.Option(None, "--model", "-m", help="Model(s) to use (repeatable)"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge"),
    output_json: bool = typer.Option(False, "--json", help="Output full report as JSON"),
    list_models: bool = typer.Option(False, "--list-models", help="Show available local models and exit"),
):
    cfg = get_config()
    client = OllamaClient(cfg)

    if list_models:
        health = client.health_check()
        if health["status"] == "offline":
            console.print(f"[red]Ollama is offline:[/red] {health.get('error', '')}")
            raise typer.Exit(1)
        console.print(f"\n[green]Ollama is online[/green] at {health['base_url']}")
        console.print(f"Found [bold]{health['model_count']}[/bold] local model(s):\n")
        for m in health["models"]:
            console.print(f"  • {m}")
        raise typer.Exit(0)

    # Load prompt text
    def load(path_or_text: str) -> str:
        if os.path.isfile(path_or_text):
            with open(path_or_text) as f:
                return f.read().strip()
        return path_or_text.strip()

    prompt_v1 = load(v1)
    prompt_v2 = load(v2)

    # Resolve models
    available = client.list_local_models()
    if not available:
        console.print("[red]No local Ollama models found. Run: ollama pull llama3[/red]")
        raise typer.Exit(1)

    selected = models if models else available[:3]
    invalid = [m for m in selected if not client.is_model_available(m)]
    if invalid:
        console.print(f"[yellow]Warning: models not found locally: {invalid}[/yellow]")
        selected = [m for m in selected if m not in invalid]
    if not selected:
        console.print("[red]No valid models selected.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running diff[/bold] across: {', '.join(selected)}\n")

    with console.status("Running pipeline..."):
        report = run_diff(
            prompt_v1=prompt_v1,
            prompt_v2=prompt_v2,
            models=selected,
            run_judge=not no_judge,
        )

    if output_json:
        print(json.dumps(report.model_dump(), indent=2))
        return

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------
    ps = report.prompt_semantic.prompt_score
    style = LEVEL_STYLES.get(ps.level, "white")

    console.print(Panel(
        f"[{style}][bold]{ps.level.value.upper()} SHIFT[/bold][/{style}]\n"
        f"{ps.label}\n"
        f"Similarity: [bold]{ps.similarity:.3f}[/bold]  |  "
        f"Shift score: [bold]{ps.shift_score:.3f}[/bold]\n"
        f"Embedding: {report.embedding_model}",
        title="🧠 Prompt Semantic Score",
        border_style=style,
    ))

    # Anatomy changes
    if report.structural_diff.anatomy_changes:
        table = Table(title="📐 Anatomy Changes", show_header=True)
        table.add_column("Type", style="bold")
        table.add_column("Change")
        table.add_column("Text", overflow="fold")
        for ch in report.structural_diff.anatomy_changes[:12]:
            tag = ch["tag"]
            color = "green" if "added" in tag else "red" if "removed" in tag else "yellow"
            table.add_row(ch["type"], f"[{color}]{tag}[/{color}]", ch["text"][:80])
        console.print(table)

    # Per-model results
    table2 = Table(title="🤖 Per-model Output Shift", show_header=True)
    table2.add_column("Model")
    table2.add_column("Shift level")
    table2.add_column("Similarity")
    table2.add_column("Shift score")
    table2.add_column("v1 latency")
    table2.add_column("v2 latency")

    for res in report.output_results:
        s = LEVEL_STYLES.get(res.score.level, "white")
        table2.add_row(
            res.model_name,
            f"[{s}]{res.score.level.value}[/{s}]",
            f"{res.score.similarity:.3f}",
            f"{res.score.shift_score:.3f}",
            f"{res.latency_v1_ms:.0f}ms",
            f"{res.latency_v2_ms:.0f}ms",
        )
    console.print(table2)

    # Judge verdict
    if report.verdict:
        v = report.verdict
        console.print(Panel(
            f"[bold]Intent change:[/bold] {v.intent_change}\n"
            f"[green][bold]Gained:[/bold][/green] {v.gained}\n"
            f"[yellow][bold]Lost:[/bold][/yellow] {v.lost}\n"
            f"[bold]Recommendation:[/bold] {v.recommendation}",
            title=f"⚖️  Judge verdict ({v.judge_model})",
            border_style="blue",
        ))


if __name__ == "__main__":
    app()
