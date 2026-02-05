from __future__ import annotations

import typer
from rich.console import Console

from . import pipeline_wizard as pipeline_wizard
from . import setup_wizard as setup_wizard

app = typer.Typer(add_completion=False, help="Unified GlossAPI entrypoint.")
console = Console()

app.add_typer(pipeline_wizard.app, name="pipeline", help="Run the GlossAPI pipeline wizard and phases.")
app.add_typer(setup_wizard.app, name="setup", help="Provision GlossAPI environments and dependencies.")


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
