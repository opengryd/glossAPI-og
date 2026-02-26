from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import shutil
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ._cli_utils import (
    _run_gum_raw,
    _open_tty,
    _simple_select,
    _simple_confirm,
    _simple_text,
    _gum_choose_impl,
    _gum_confirm_impl,
    _gum_input_impl,
)

app = typer.Typer(add_completion=False, help="Environment setup wizard for GlossAPI.")
console = Console()

_ALL_MODES = ["vanilla", "rapidocr", "mineru", "deepseek-ocr", "deepseek-ocr-2", "glm-ocr", "olmocr"]

# Mode labels shown in interactive menus.  Descriptions are displayed in the
# banner table inside glossapi-cli.sh; platform-incompatibility warnings are
# emitted by _ask_mode() after the user selects a mode.
_MODE_LABELS: dict[str, dict[str, str]] = {
    "vanilla":        {"Darwin": "vanilla",
                       "Linux":  "vanilla"},
    "rapidocr":       {"Darwin": "rapidocr",
                       "Linux":  "rapidocr"},
    "mineru":         {"Darwin": "mineru",
                       "Linux":  "mineru"},
    "deepseek-ocr":   {"Darwin": "deepseek-ocr",
                       "Linux":  "deepseek-ocr"},
    "deepseek-ocr-2": {"Darwin": "deepseek-ocr-2",
                       "Linux":  "deepseek-ocr-2"},
    "glm-ocr":        {"Darwin": "glm-ocr",
                       "Linux":  "glm-ocr"},
    "olmocr":         {"Darwin": "olmocr",
                       "Linux":  "olmocr"},
}


def _mode_choices_for_platform() -> list[str]:
    """Return annotated mode labels appropriate for the current platform."""
    system = platform.system()
    return [
        _MODE_LABELS.get(m, {}).get(system, m)
        for m in _ALL_MODES
    ]


def _label_to_mode(label: str) -> str:
    """Extract the bare mode name from an annotated label like 'rapidocr (Docling ...)'."""
    return label.split()[0].rstrip()


MODE_CHOICES = _ALL_MODES  # kept for backward compat (--mode flag validation)
SIMPLE_PROMPTS = os.environ.get("GLOSSAPI_SETUP_SIMPLE", "0") == "1"


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and not SIMPLE_PROMPTS


def _available_python_versions() -> list[str]:
    versions: list[str] = []
    for cmd in ("python3.11", "python3.12", "python3.13"):
        if shutil.which(cmd):
            versions.append(cmd)
    return versions


def _require_gum() -> None:
    if shutil.which("gum") is None:
        console.print("[red]gum is required for interactive prompts. Install it and retry.[/red]")
        raise typer.Exit(code=1)


def _run_gum(args: list[str]) -> tuple[int, str]:
    _require_gum()
    return _run_gum_raw(args)


def _gum_choose(label: str, choices: list[str], *, default: Optional[str] = None) -> list[str]:
    """Thin wrapper binding the wizard's ``_run_gum`` to the shared impl.

    Always returns a list (single-select callers index with ``[0]``).
    Raises ``typer.Exit`` on user cancel (empty result).
    """
    result = _gum_choose_impl(_run_gum, label, choices, default=default)
    if not result:
        raise typer.Exit(code=1)
    return result


def _gum_confirm(label: str, default: bool) -> bool:
    return _gum_confirm_impl(_run_gum, label, default)


def _gum_input(label: str, default: str) -> str:
    result = _gum_input_impl(_run_gum, label, default)
    if result is None:
        raise typer.Exit(code=1)
    return result


# _open_tty, _simple_select, _simple_confirm, _simple_text imported from ._cli_utils


def _detect_os_label() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macOS Apple Silicon (MPS/Metal + MLX supported)"
    if system == "Linux":
        return "Linux (CUDA supported)"
    if system == "Windows":
        return "Windows (CUDA supported)"
    return system


def _ask_mode(default: Optional[str]) -> str:
    if default:
        return default
    system = platform.system()
    default_mode = "mineru" if system == "Darwin" else "rapidocr"
    choices = _mode_choices_for_platform()
    # Find the annotated default label
    default_choice = next((c for c in choices if c.startswith(default_mode)), choices[0])
    if _is_interactive():
        selected = _gum_choose("Environment profile", choices, default=default_choice)[0]
    else:
        tty = _open_tty()
        if tty:
            with tty:
                selected = _simple_select("Environment profile", choices, default_choice, tty)
        else:
            selected = _simple_select("Environment profile", choices, default_choice, None)
    mode = _label_to_mode(selected)
    # Warn about platform-incompatible selections
    if system == "Darwin" and mode == "deepseek-ocr":
        console.print(
            "[bold yellow]⚠️  WARNING:[/bold yellow] deepseek-ocr requires CUDA + vLLM "
            "which are [bold]NOT available on macOS[/bold].\n"
            "   Consider [green]deepseek-ocr-2[/green] (MLX/MPS) or "
            "[green]glm-ocr[/green] instead."
        )
        if not _ask_bool("Continue with deepseek-ocr anyway?", default=False):
            raise typer.Exit(code=0)
    if system == "Linux" and mode in {"deepseek-ocr-2", "glm-ocr"}:
        console.print(
            f"[bold yellow]⚠️  WARNING:[/bold yellow] {mode} uses MLX which is "
            "[bold]only available on macOS/Apple Silicon[/bold].\n"
            "   Consider [green]deepseek-ocr[/green] (CUDA/vLLM) or "
            "[green]olmocr[/green] instead."
        )
        if not _ask_bool(f"Continue with {mode} anyway?", default=False):
            raise typer.Exit(code=0)
    return mode


def _ask_venv(default: Optional[str], *, mode: str = "") -> Path:
    if default:
        return Path(default).expanduser().resolve()
    # Default venv name is derived from the chosen mode so each backend gets
    # its own isolated environment (e.g. .venvs/rapidocr, .venvs/deepseek-ocr).
    venv_name = mode if mode else "glossapi"
    default_path = str(Path("dependency_setup") / ".venvs" / venv_name)
    if _is_interactive():
        response = _gum_input("Virtualenv path", default_path)
        if not response:
            raise typer.Exit(code=1)
        return Path(response).expanduser().resolve()
    tty = _open_tty()
    if tty:
        with tty:
            response = _simple_text("Virtualenv path", default_path, tty)
            return Path(response).expanduser().resolve()
    response = _simple_text("Virtualenv path", default_path, None)
    return Path(response).expanduser().resolve()


def _ask_python(default: Optional[str]) -> str:
    if default:
        return default
    versions = _available_python_versions()
    if not versions:
        console.print("[red]Python 3.11–3.13 not found on PATH. Install one and retry.[/red]")
        raise typer.Exit(code=1)
    if _is_interactive():
        return _gum_choose("Python version", versions, default=versions[0])[0]
    tty = _open_tty()
    if tty:
        with tty:
            return _simple_select("Python version", versions, versions[0], tty)
    return _simple_select("Python version", versions, versions[0], None)


def _ask_bool(label: str, default: bool = False) -> bool:
    if _is_interactive():
        return _gum_confirm(label, default)
    tty = _open_tty()
    if tty:
        with tty:
            return _simple_confirm(label, default, tty)
    return _simple_confirm(label, default, None)


def _run_setup(
    mode: str,
    venv: Path,
    *,
    python_bin: str,
    download_deepseek_ocr: bool,
    download_deepseek_ocr2: bool,
    download_glmocr: bool,
    download_olmocr: bool,
    weights_root: Optional[str],
    download_mineru: bool,
    detectron2_auto_install: bool,
    detectron2_wheel_url: Optional[str],
) -> bool:
    script = Path("dependency_setup") / "setup_glossapi.sh"
    if not script.exists():
        console.print(f"[red]Missing setup script: {script}[/red]")
        raise typer.Exit(code=1)

    args = ["bash", str(script), "--mode", mode, "--venv", str(venv), "--python", python_bin]
    if download_deepseek_ocr:
        args.append("--download-deepseek-ocr")
    if download_deepseek_ocr2:
        args.append("--download-deepseek-ocr2")
    if download_glmocr:
        args.append("--download-glmocr")
    if download_olmocr:
        args.append("--download-olmocr")
    if weights_root:
        args.extend(["--weights-root", weights_root])
    if download_mineru:
        args.append("--download-mineru-models")

    env = os.environ.copy()
    if detectron2_wheel_url:
        env["DETECTRON2_WHL_URL"] = detectron2_wheel_url
    if detectron2_auto_install:
        env["DETECTRON2_AUTO_INSTALL"] = "1"

    console.print(Panel.fit("Running setup", title="GlossAPI"))
    result = subprocess.run(args, env=env)
    if result.returncode == 0:
        weights_downloaded: list[str] = []
        if download_deepseek_ocr:
            weights_downloaded.append("deepseek-ocr")
        if download_deepseek_ocr2:
            weights_downloaded.append("deepseek-ocr-2")
        if download_glmocr:
            weights_downloaded.append("glm-ocr")
        if download_olmocr:
            weights_downloaded.append("olmocr")
        if download_mineru:
            weights_downloaded.append("mineru")
        from . import _state  # late import to avoid circular deps at module load
        _state.record_setup(
            mode,
            venv,
            python_bin=python_bin,
            weights_downloaded=weights_downloaded or None,
        )
        console.print(
            f"[green]✓ Backend '{mode}' recorded in state.[/green] "
            f"[dim](dependency_setup/.glossapi_state.json)[/dim]"
        )
        console.print("[dim]  Run [bold]glossapi status[/bold] to view all installed backends.[/dim]")
        return True
    return False


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    mode: Optional[str] = typer.Option(None, help="Profile: vanilla/rapidocr/mineru/deepseek-ocr/deepseek-ocr-2/glm-ocr/olmocr."),
    venv: Optional[str] = typer.Option(None, help="Path to the target virtualenv."),
    python: Optional[str] = typer.Option(None, help="Python executable to use (3.11–3.13)."),
    download_deepseek_ocr: bool = typer.Option(False, help="Download DeepSeek OCR weights."),
    download_deepseek_ocr2: bool = typer.Option(False, help="Download DeepSeek OCR v2 weights."),
    download_glmocr: bool = typer.Option(False, help="Download GLM-OCR MLX weights."),
    download_olmocr: bool = typer.Option(False, help="Download OlmOCR weights."),
    weights_root: Optional[str] = typer.Option(None, help="Root directory for all model weights."),
    download_mineru_models: bool = typer.Option(False, help="Download MinerU models."),
    detectron2_auto_install: bool = typer.Option(False, help="Auto-install detectron2 when using MinerU (macOS arm64)."),
    detectron2_wheel_url: Optional[str] = typer.Option(None, help="Detectron2 wheel URL for MinerU."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    console.print(Panel.fit("GlossAPI Setup", title="Welcome"))
    console.print(f"[dim]Platform: {_detect_os_label()}[/dim]")

    # 1. Profile — drives defaults for every subsequent question.
    selected_mode = _ask_mode(mode)

    # 2. Virtualenv — default name derived from profile so each backend is isolated.
    selected_venv = _ask_venv(venv, mode=selected_mode)

    # 3. Python version.
    selected_python = _ask_python(python)

    # 4. Optional model-weight downloads (only shown for relevant profiles).
    if selected_mode == "deepseek-ocr" and not download_deepseek_ocr:
        download_deepseek_ocr = _ask_bool(
            "Download DeepSeek OCR weights now? (skip for faster setup)",
            default=False,
        )
    if selected_mode == "deepseek-ocr-2" and not download_deepseek_ocr2:
        download_deepseek_ocr2 = _ask_bool(
            "Download DeepSeek OCR v2 weights now? (skip to auto-download at runtime)",
            default=False,
        )
    if selected_mode == "glm-ocr" and not download_glmocr:
        download_glmocr = _ask_bool(
            "Download GLM-OCR MLX weights now? (skip to auto-download at runtime)",
            default=False,
        )
    if selected_mode == "olmocr" and not download_olmocr:
        download_olmocr = _ask_bool(
            "Download OlmOCR weights now? (skip to auto-download at runtime)",
            default=False,
        )
    if selected_mode == "mineru" and not download_mineru_models:
        download_mineru_models = _ask_bool(
            "Download MinerU models now? (large download)",
            default=False,
        )

    # 5. detectron2 (MinerU only) — single choice instead of two separate prompts.
    if selected_mode == "mineru" and not detectron2_wheel_url and not detectron2_auto_install:
        d2_choices = ["auto-install from source", "provide wheel URL", "skip"]
        if _is_interactive():
            d2_choice = _gum_choose("detectron2 setup", d2_choices, default=d2_choices[0])[0]
        else:
            tty = _open_tty()
            if tty:
                with tty:
                    d2_choice = _simple_select("detectron2 setup", d2_choices, d2_choices[0], tty)
            else:
                d2_choice = _simple_select("detectron2 setup", d2_choices, d2_choices[0], None)
        if d2_choice == "provide wheel URL":
            if _is_interactive():
                detectron2_wheel_url = _gum_input("Wheel URL", "") or None
            else:
                tty = _open_tty()
                if tty:
                    with tty:
                        detectron2_wheel_url = _simple_text("Detectron2 wheel URL", "", tty) or None
                else:
                    detectron2_wheel_url = _simple_text("Detectron2 wheel URL", "", None) or None
        elif d2_choice == "auto-install from source":
            detectron2_auto_install = True

    # 6. Summary panel — show everything before the script runs.
    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_row("Profile", f"[green]{selected_mode}[/green]")
    summary.add_row("Virtualenv", str(selected_venv))
    summary.add_row("Python", selected_python)
    weights_to_dl = [
        b for b, flag in [
            ("deepseek-ocr", download_deepseek_ocr),
            ("deepseek-ocr-2", download_deepseek_ocr2),
            ("glm-ocr", download_glmocr),
            ("olmocr", download_olmocr),
            ("mineru", download_mineru_models),
        ] if flag
    ]
    if weights_to_dl:
        summary.add_row("Download weights", ", ".join(weights_to_dl))
    console.print(Panel(summary, title="Setup summary"))
    if not _ask_bool("Proceed with setup?", default=True):
        raise typer.Exit(code=0)

    success = _run_setup(
        selected_mode,
        selected_venv,
        python_bin=selected_python,
        download_deepseek_ocr=download_deepseek_ocr,
        download_deepseek_ocr2=download_deepseek_ocr2,
        download_glmocr=download_glmocr,
        download_olmocr=download_olmocr,
        weights_root=weights_root,
        download_mineru=download_mineru_models,
        detectron2_auto_install=detectron2_auto_install,
        detectron2_wheel_url=detectron2_wheel_url or None,
    )

    if not success:
        raise typer.Exit(code=1)

    # 7. Handoff — offer to jump straight into a pipeline run with the just-installed backend
    # pre-selected so the user doesn't have to re-pick it from the menu.
    if _ask_bool("Run pipeline now?", default=True):
        from .pipeline_wizard import _run_wizard
        _run_wizard(
            input_dir=None,
            output_dir=None,
            input_format=None,
            phase=None,
            confirm_each=False,
            log_level="INFO",
            default_backend=selected_mode,
        )
    raise typer.Exit(code=0)
