from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import pipeline_wizard as pipeline_wizard
from . import setup_wizard as setup_wizard

app = typer.Typer(add_completion=False, help="Unified GlossAPI entrypoint.")
console = Console()

app.add_typer(pipeline_wizard.app, name="pipeline", help="Run the GlossAPI pipeline wizard and phases.")
app.add_typer(setup_wizard.app, name="setup", help="Provision GlossAPI environments and dependencies.")


# ---------------------------------------------------------------------------
# glossapi status
# ---------------------------------------------------------------------------

_ALL_BACKENDS = [
    "vanilla",
    "rapidocr",
    "mineru",
    "deepseek-ocr",
    "deepseek-ocr-2",
    "glm-ocr",
    "olmocr",
]

# Backends whose weights are large / optional downloads.
_WEIGHT_BACKENDS = {"deepseek-ocr", "deepseek-ocr-2", "glm-ocr", "olmocr", "mineru"}


@app.command("status")
def status() -> None:
    """Show which GlossAPI backends are set up and available.

    Reads the state file written by ``glossapi setup`` to display a summary
    table of installed backends, their virtual-environment paths, Python
    versions, and whether model weights have been downloaded.
    """
    from . import _state

    data = _state.all_state()
    backends_info = data.get("backends", {})
    weights_info = data.get("weights", {})
    last_updated = data.get("last_updated")

    console.print(Panel.fit("GlossAPI Backend Status", title="glossapi status"))
    console.print(f"[dim]Platform: {platform.system()} {platform.machine()}[/dim]")
    if last_updated:
        console.print(f"[dim]State last updated: {last_updated[:19]} UTC[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Backend", style="bold", min_width=16)
    table.add_column("Status", min_width=14)
    table.add_column("Venv path", overflow="fold", min_width=20)
    table.add_column("Python", min_width=12)
    table.add_column("Weights")
    table.add_column("Installed")

    for b in _ALL_BACKENDS:
        info = backends_info.get(b, {})
        if info.get("installed"):
            status_str = "[green]✓ ready[/green]"
            venv_str = info.get("venv", "—")
            python_str = info.get("python", "—")
            installed_str = (info.get("installed_at") or "")[:10] or "—"
        else:
            status_str = "[dim]✗ not set up[/dim]"
            venv_str = "[dim]—[/dim]"
            python_str = "[dim]—[/dim]"
            installed_str = "[dim]—[/dim]"

        if b in _WEIGHT_BACKENDS:
            weights_str = (
                "[green]✓[/green]" if weights_info.get(b) else "[dim]—[/dim]"
            )
        else:
            weights_str = "[dim]n/a[/dim]"

        table.add_row(b, status_str, venv_str, python_str, weights_str, installed_str)

    console.print(table)
    console.print()

    ready = [b for b in _ALL_BACKENDS if backends_info.get(b, {}).get("installed")]
    if not ready:
        if not _state.has_state_file():
            console.print(
                "[yellow]No state file found.[/yellow] "
                "Run [bold]glossapi setup[/bold] to install a backend first."
            )
        else:
            console.print(
                "[yellow]No backends installed yet.[/yellow] "
                "Run [bold]glossapi setup[/bold] to install one."
            )
    else:
        console.print(
            f"[green]{len(ready)} backend(s) ready:[/green] "
            + ", ".join(f"[bold]{b}[/bold]" for b in ready)
        )
        console.print(
            "[dim]Run [bold]glossapi pipeline[/bold] to start processing.[/dim]"
        )


# ---------------------------------------------------------------------------
# Default: launch pipeline wizard
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    console.print("[dim]No subcommand provided; launching the pipeline wizard.[/dim]")
    pipeline_wizard._run_wizard(
        input_dir=None,
        output_dir=None,
        input_format=None,
        phase=None,
        confirm_each=True,
        log_level="INFO",
    )
