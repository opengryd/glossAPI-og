from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import shutil
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False, help="Environment setup wizard for GlossAPI.")
console = Console()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

MODE_CHOICES = ["vanilla", "rapidocr", "mineru", "deepseek", "deepseek-ocr-2"]
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
    try:
        tty_in = open("/dev/tty", "rb")
        tty_out = open("/dev/tty", "wb")
    except Exception:
        proc = subprocess.run(["gum", *args], text=True, capture_output=True)
        output = _ANSI_RE.sub("", proc.stdout or "")
        return proc.returncode, output.strip()

    proc = subprocess.run(
        ["gum", *args],
        stdin=tty_in,
        stdout=subprocess.PIPE,
        stderr=tty_out,
        text=True,
    )
    tty_in.close()
    tty_out.close()
    output = _ANSI_RE.sub("", proc.stdout or "")
    return proc.returncode, output.strip()


def _gum_choose(label: str, choices: list[str], default: Optional[str]) -> str:
    args = ["choose", "--header", label]
    if default:
        args.extend(["--selected", default])
    args.extend(choices)
    code, output = _run_gum(args)
    if code != 0:
        raise typer.Exit(code=1)
    return output


def _gum_confirm(label: str, default: bool) -> bool:
    args = ["confirm", "--default=true" if default else "--default=false", label]
    code, _ = _run_gum(args)
    return code == 0


def _gum_input(label: str, default: str) -> str:
    args = ["input", "--prompt", f"{label}: ", "--value", default]
    code, output = _run_gum(args)
    if code != 0:
        raise typer.Exit(code=1)
    return output


def _open_tty() -> Optional[object]:
    try:
        return open("/dev/tty", "r+")
    except Exception:
        return None


def _simple_select(label: str, choices: list[str], default: str, tty: Optional[object]) -> str:
    default_idx = choices.index(default) + 1 if default in choices else 1
    def _write(msg: str) -> None:
        if tty:
            tty.write(msg)
            tty.flush()
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def _readline() -> str:
        if tty:
            return tty.readline()
        return input()

    _write(f"{label}:\n")
    for idx, choice in enumerate(choices, start=1):
        _write(f"  {idx}) {choice}\n")
    while True:
        _write(f"Select [default {default_idx}]: ")
        answer = _readline().strip()
        if not answer:
            return choices[default_idx - 1]
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        _write("Invalid choice. Try again.\n")


def _simple_confirm(label: str, default: bool, tty: Optional[object]) -> bool:
    default_char = "Y" if default else "n"
    while True:
        if tty:
            tty.write(f"{label} [Y/n] (default {default_char}): ")
            tty.flush()
            answer = tty.readline().strip().lower()
        else:
            answer = input(f"{label} [Y/n] (default {default_char}): ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        if tty:
            tty.write("Please answer y or n.\n")
            tty.flush()
        else:
            print("Please answer y or n.")


def _simple_text(label: str, default: str, tty: Optional[object]) -> str:
    if tty:
        tty.write(f"{label} [default: {default}]: ")
        tty.flush()
        answer = tty.readline().strip()
    else:
        answer = input(f"{label} [default: {default}]: ").strip()
    return answer or default


def _detect_os_label() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macOS (MPS/Metal supported)"
    if system == "Linux":
        return "Linux (CUDA supported)"
    if system == "Windows":
        return "Windows (CUDA supported)"
    return system


def _ask_mode(default: Optional[str]) -> str:
    if default:
        return default
    default_mode = "vanilla"
    choices = MODE_CHOICES
    default_choice = default_mode
    if _is_interactive():
        return _gum_choose("Environment profile", choices, default_choice)
    tty = _open_tty()
    if tty:
        with tty:
            return _simple_select("Environment profile", choices, default_choice, tty)
    return _simple_select("Environment profile", choices, default_choice, None)


def _ask_venv(default: Optional[str]) -> Path:
    if default:
        return Path(default).expanduser().resolve()
    default_path = str(Path("dependency_setup") / ".venvs" / "glossapi")
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
        return _gum_choose("Python version", versions, versions[0])
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
    download_deepseek: bool,
    weights_dir: Optional[str],
    download_deepseek_ocr2: bool,
    download_mineru: bool,
    run_tests: bool,
    smoke_test: bool,
    detectron2_auto_install: bool,
    detectron2_wheel_url: Optional[str],
) -> None:
    script = Path("dependency_setup") / "setup_glossapi.sh"
    if not script.exists():
        console.print(f"[red]Missing setup script: {script}[/red]")
        raise typer.Exit(code=1)

    args = ["bash", str(script), "--mode", mode, "--venv", str(venv), "--python", python_bin]
    if download_deepseek:
        args.append("--download-deepseek")
    if download_deepseek_ocr2:
        args.append("--download-deepseek-ocr2")
    if weights_dir:
        args.extend(["--weights-dir", weights_dir])
    if download_mineru:
        args.append("--download-mineru-models")
    if run_tests:
        args.append("--run-tests")
    if smoke_test:
        args.append("--smoke-test")

    env = os.environ.copy()
    if detectron2_wheel_url:
        env["DETECTRON2_WHL_URL"] = detectron2_wheel_url
    if detectron2_auto_install:
        env["DETECTRON2_AUTO_INSTALL"] = "1"

    console.print(Panel.fit("Running setup", title="GlossAPI"))
    raise typer.Exit(code=subprocess.run(args, env=env).returncode)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    mode: Optional[str] = typer.Option(None, help="Profile: vanilla/rapidocr/mineru/deepseek/deepseek-ocr-2."),
    venv: Optional[str] = typer.Option(None, help="Path to the target virtualenv."),
    python: Optional[str] = typer.Option(None, help="Python executable to use (3.11–3.13)."),
    download_deepseek: bool = typer.Option(False, help="Download DeepSeek weights."),
    weights_dir: Optional[str] = typer.Option(None, help="Weights dir for DeepSeek."),
    download_deepseek_ocr2: bool = typer.Option(False, help="Download DeepSeek OCR v2 weights."),
    weights_dir_ocr2: Optional[str] = typer.Option(None, help="Weights dir for DeepSeek OCR v2."),
    download_mineru_models: bool = typer.Option(False, help="Download MinerU models."),
    detectron2_auto_install: bool = typer.Option(False, help="Auto-install detectron2 when using MinerU (macOS arm64)."),
    detectron2_wheel_url: Optional[str] = typer.Option(None, help="Detectron2 wheel URL for MinerU."),
    run_tests: bool = typer.Option(False, help="Run profile-appropriate tests."),
    smoke_test: bool = typer.Option(False, help="Run the DeepSeek smoke test."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    console.print(Panel.fit("GlossAPI Setup", title="Welcome"))
    console.print(f"[dim]Detected OS: {_detect_os_label()}[/dim]")
    selected_mode = _ask_mode(mode)
    selected_venv = _ask_venv(venv)
    selected_python = _ask_python(python)

    if selected_mode == "deepseek" and not download_deepseek:
        download_deepseek = _ask_bool(
            "Download DeepSeek weights now? (skip for faster setup)",
            default=False,
        )
    if selected_mode == "deepseek-ocr-2" and not download_deepseek_ocr2:
        download_deepseek_ocr2 = _ask_bool(
            "Download DeepSeek OCR v2 weights now? (skip to auto-download at runtime)",
            default=False,
        )
    if selected_mode == "mineru" and not download_mineru_models:
        download_mineru_models = _ask_bool(
            "Download MinerU models now? (large download)",
            default=False,
        )
    if selected_mode == "mineru" and not detectron2_wheel_url:
        if _is_interactive():
            detectron2_wheel_url = _gum_input(
                "Detectron2 wheel URL (optional)",
                detectron2_wheel_url or "",
            )
        else:
            tty = _open_tty()
            if tty:
                with tty:
                    detectron2_wheel_url = _simple_text(
                        "Detectron2 wheel URL (optional)",
                        detectron2_wheel_url or "",
                        tty,
                    )
            else:
                detectron2_wheel_url = _simple_text(
                    "Detectron2 wheel URL (optional)",
                    detectron2_wheel_url or "",
                    None,
                )
        if detectron2_wheel_url:
            detectron2_auto_install = False
    if selected_mode == "mineru" and not detectron2_auto_install and not detectron2_wheel_url:
        detectron2_auto_install = _ask_bool(
            "Attempt detectron2 auto-install from source? (slow; use if no wheel/preinstall)",
            default=True,
        )
    if not run_tests:
        run_tests = _ask_bool("Run tests after setup? (slower)", default=False)
    if selected_mode == "deepseek" and not smoke_test:
        smoke_test = _ask_bool("Run DeepSeek smoke test? (requires GPU)", default=False)

    _run_setup(
        selected_mode,
        selected_venv,
        python_bin=selected_python,
        download_deepseek=download_deepseek,
        download_deepseek_ocr2=download_deepseek_ocr2,
        weights_dir=weights_dir,
        download_mineru=download_mineru_models,
        run_tests=run_tests,
        smoke_test=smoke_test,
        detectron2_auto_install=detectron2_auto_install,
        detectron2_wheel_url=detectron2_wheel_url or None,
    )
