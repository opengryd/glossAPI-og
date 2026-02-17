from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import importlib.util
import typer
from rich.console import Console
from rich.panel import Panel

from .corpus.corpus_orchestrator import Corpus

app = typer.Typer(add_completion=False, help="Interactive wizard for the GlossAPI pipeline.")
console = Console()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PHASE_CHOICES = [
    "extract",
    "clean",
    "ocr",
    "section",
    "annotate",
    "export-jsonl",
]

PRESET_CHOICES = [
    "Lightweight PDF smoke test",
    "MinerU demo (samples/eellak)",
    "Custom",
]

ACCEL_CHOICES = [
    "CPU",
    "GPU (auto)",
]


def _require_gum() -> None:
    if shutil.which("gum") is None:
        console.print("[red]gum is required for interactive prompts. Install it and retry.[/red]")
        raise typer.Exit(code=1)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print("[red]Interactive CLI requires a TTY. Run in a terminal session.[/red]")
        raise typer.Exit(code=1)


def _run_gum(args: List[str]) -> tuple[int, str]:
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


def _gum_choose(
    label: str,
    choices: List[str],
    *,
    default: Optional[List[str] | str] = None,
    multi: bool = False,
) -> List[str]:
    args = ["choose", "--header", label]
    if multi:
        args.append("--no-limit")
    if default:
        if isinstance(default, list):
            for item in default:
                args.extend(["--selected", item])
        else:
            args.extend(["--selected", default])
    args.extend(choices)
    code, output = _run_gum(args)
    if code != 0:
        raise typer.Exit(code=1)
    if multi:
        return [line for line in output.splitlines() if line.strip()]
    return [output] if output else []


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


def _ensure_path(
    value: Optional[str],
    *,
    label: str,
    default: Optional[Path] = None,
    must_exist: bool = False,
) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    fallback = default or Path.cwd()
    while True:
        response = _gum_input(label, str(fallback))
        if not response:
            raise typer.Exit(code=1)
        candidate = Path(response).expanduser().resolve()
        if not must_exist:
            return candidate
        if candidate.exists() and candidate.is_dir():
            return candidate
        console.print(f"[yellow]Path not found or not a directory: {candidate}.[/yellow]")


def _normalize_input_format(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"markdown", "md"}:
        return "md"
    return lowered


def _ask_input_format(value: Optional[str]) -> str:
    if value:
        return _normalize_input_format(value)
    choice = _gum_choose("Input format", ["pdf", "markdown"], default="pdf")
    return _normalize_input_format(choice[0])


def _ask_preset() -> str:
    choice = _gum_choose("Workflow preset", PRESET_CHOICES, default="Custom")
    return choice[0]


def _detect_os_label() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macOS (MPS/Metal supported)"
    if system == "Linux":
        return "Linux (CUDA supported)"
    if system == "Windows":
        return "Windows (CUDA supported)"
    return system


def _ask_accel_mode() -> str:
    choice = _gum_choose("Acceleration", ACCEL_CHOICES, default="GPU (auto)")
    return choice[0]


def _ensure_input_has_files(input_dir: Path, input_format: str) -> Path:
    normalized = _normalize_input_format(input_format)
    patterns = ["*.pdf"] if normalized == "pdf" else ["*.md", "*.markdown"]
    while True:
        has_files = any(input_dir.glob(pattern) for pattern in patterns)
        if has_files:
            return input_dir
        console.print(
            f"[yellow]No {input_format} files found in {input_dir} (top-level only)." "[/yellow]"
        )
        retry = _gum_confirm("Pick another input directory?", default=True)
        if not retry:
            return input_dir
        input_dir = _ensure_path(None, label="Input directory", default=Path.cwd())


def _ask_phases(phases: Optional[Iterable[str]]) -> List[str]:
    if phases:
        return list(phases)
    response = _gum_choose("Select phases to run", PHASE_CHOICES, default=PHASE_CHOICES, multi=True)
    if not response:
        raise typer.Exit(code=1)
    return list(response)


def _default_input_dir(input_format: str) -> Path:
    normalized = _normalize_input_format(input_format)
    if normalized == "pdf":
        return Path("samples") / "lightweight_pdf_corpus" / "pdfs"
    return Path("samples")


def _ask_ocr_backend() -> str:
    default_backend = "mineru" if platform.system() == "Darwin" else "rapidocr"
    choice = _gum_choose(
        "OCR backend",
        ["none", "rapidocr", "mineru", "deepseek-ocr", "deepseek-ocr-2", "glm-ocr", "olmocr"],
        default=default_backend,
    )
    return choice[0]


def _maybe_confirm(phase: str, *, confirm_each: bool) -> bool:
    if not confirm_each:
        return True
    return _gum_confirm(f"Run phase: {phase}?", default=True)


# Maps CUDA-dependent backend names to the env var that configures their
# separate Python binary (if any).  When the env var is set, the wizard
# probes that binary for ``torch.cuda.is_available()`` before launching
# the pipeline.
_CUDA_BACKEND_PYTHON_ENV: dict = {
    "olmocr": "GLOSSAPI_OLMOCR_PYTHON",
    "deepseek-ocr": "GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON",
    # MinerU uses magic-pdf CLI, not a Python binary — checked separately.
}

# Human-readable labels for each backend.
_BACKEND_LABELS: dict = {
    "olmocr": "OlmOCR",
    "deepseek-ocr": "DeepSeek-OCR",
    "mineru": "MinerU",
}


def _cuda_preflight(backend: str) -> None:
    """Pre-flight check: verify CUDA is available for a CUDA-dependent backend.

    Checks three things in order:

    1. If the backend has a configured subprocess Python binary (via its env
       var), probe it with ``torch.cuda.is_available()``.
    2. If no separate binary is configured, check the *current* process for
       CUDA availability (in case the user runs everything in one venv).
    3. If CUDA is not available, print actionable diagnostics and exit.
    """
    from glossapi.ocr.utils.cuda import probe_cuda

    label = _BACKEND_LABELS.get(backend, backend)
    env_var = _CUDA_BACKEND_PYTHON_ENV.get(backend)
    stub_env = f"GLOSSAPI_{backend.upper().replace('-', '_')}_ALLOW_STUB"

    # Determine the Python binary to probe.
    env_python = os.environ.get(env_var, "").strip() if env_var else ""
    if env_python:
        python_bin = Path(env_python)
        if not python_bin.exists():
            console.print(f"[red]{label} Python binary not found: {python_bin}[/red]")
            console.print(f"Check {env_var} and verify the path exists.")
            raise typer.Exit(code=1)
    else:
        # No separate venv; probe current interpreter as a sanity check.
        python_bin = Path(sys.executable)

    cuda_ok = probe_cuda(python_bin)
    if cuda_ok is True:
        return
    if cuda_ok is None:
        # Could not determine — don't block; let the runner handle it.
        return

    # CUDA is explicitly False.
    console.print(f"[red]CUDA is not available in the {label} Python environment.[/red]")
    console.print(f"[dim]Python binary: {python_bin}[/dim]")
    console.print("")
    console.print("[bold]Troubleshooting:[/bold]")
    console.print(f"  1. Verify CUDA in the {label} venv:")
    console.print(f"       {python_bin} -c \"import torch; print(torch.cuda.is_available())\"")
    console.print("  2. Install CUDA-enabled PyTorch if it reports False:")
    console.print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
    console.print("  3. Ensure CUDA runtime libraries are findable:")
    if env_var:
        ld_env = env_var.replace("_PYTHON", "_LD_LIBRARY_PATH").replace(
            "_TEST_PYTHON", "_LD_LIBRARY_PATH"
        )
        console.print(f"       export {ld_env}=/usr/local/cuda/lib64")
    else:
        console.print("       export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    console.print(f"  4. To use placeholder output instead:")
    console.print(f"       export {stub_env}=1")
    raise typer.Exit(code=1)


def _run_jsonl_export(corpus: Corpus, output_dir: Path, input_dir: Path) -> Optional[Path]:
    default_path = output_dir / "export" / "corpus.jsonl"
    response = _gum_input("JSONL output path", str(default_path))
    if not response:
        raise typer.Exit(code=1)
    export_path = Path(response).expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "download_results" / "download_results.parquet"
    if not metadata_path.exists():
        console.print("[yellow]Metadata parquet not found. JSONL export needs a parquet file.[/yellow]")
        fallback_default = input_dir / "download_results" / "download_results.parquet"
        response = _gum_input("Metadata parquet path (leave blank to skip export)", str(fallback_default))
        if not response:
            console.print("[yellow]Skipping JSONL export.[/yellow]")
            return None
        metadata_path = Path(response).expanduser().resolve()
        if not metadata_path.exists():
            console.print(f"[red]Metadata parquet not found at {metadata_path}. Skipping export.[/red]")
            return None

    return corpus.jsonl(export_path, metadata_path=metadata_path)


def _run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    input_format: str,
    phases: List[str],
    confirm_each: bool,
    log_level: int,
    ocr_kwargs: Optional[dict] = None,
    extract_kwargs: Optional[dict] = None,
    clean_kwargs: Optional[dict] = None,
) -> None:
    corpus = Corpus(input_dir=input_dir, output_dir=output_dir, log_level=log_level)

    def _run_extract() -> None:
        if input_format == "md":
            markdown_dir = output_dir / "markdown"
            downloads_dir = output_dir / "downloads"
            markdown_dir.mkdir(parents=True, exist_ok=True)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            candidates = list(input_dir.glob("*.md")) + list(input_dir.glob("*.markdown"))
            if not candidates:
                console.print(f"[yellow]No markdown files found in {input_dir}.[/yellow]")
                return
            for path in candidates:
                shutil.copy2(path, markdown_dir / path.name)
                shutil.copy2(path, downloads_dir / path.name)
            console.print(f"[green]Copied {len(candidates)} markdown files into {markdown_dir}.[/green]")
            return
        if extract_kwargs and extract_kwargs.get("phase1_backend") == "safe":
            previous = os.environ.get("GLOSSAPI_SKIP_DOCLING_BOOT")
            os.environ["GLOSSAPI_SKIP_DOCLING_BOOT"] = "1"
            try:
                corpus.extract(input_format=input_format, **extract_kwargs)
            finally:
                if previous is None:
                    os.environ.pop("GLOSSAPI_SKIP_DOCLING_BOOT", None)
                else:
                    os.environ["GLOSSAPI_SKIP_DOCLING_BOOT"] = previous
            return
        if importlib.util.find_spec("docling") is None:
            console.print(
                "[red]Missing dependency: docling.[/red]"
            )
            console.print(
                "Run glossapi setup and choose the RapidOCR profile to install dependencies."
            )
            console.print(
                "Or choose the 'MinerU demo' preset for the sample OCR flow."
            )
            raise typer.Exit(code=1)
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore  # noqa: F401
        except Exception:
            console.print(
                "[red]Transformers is too old for Docling (AutoModelForImageTextToText missing).[/red]"
            )
            console.print("Run glossapi setup with the RapidOCR profile to update dependencies.")
            raise typer.Exit(code=1)
        corpus.extract(input_format=input_format, **(extract_kwargs or {}))

    def _run_ocr() -> None:
        if ocr_kwargs:
            if ocr_kwargs.get("backend") == "mineru":
                env_cmd = os.environ.get("GLOSSAPI_MINERU_COMMAND")
                candidate = Path(env_cmd).expanduser() if env_cmd else None
                cmd_exists = candidate.exists() if candidate else False
                if not cmd_exists and shutil.which("magic-pdf") is None:
                    console.print("[red]Missing MinerU CLI (magic-pdf).[/red]")
                    console.print("Run glossapi setup and choose the MinerU profile to install dependencies.")
                    console.print("Set GLOSSAPI_MINERU_COMMAND or install magic-pdf, then retry.")
                    raise typer.Exit(code=1)
                try:
                    import detectron2  # type: ignore  # noqa: F401
                    has_detectron2 = True
                except Exception:
                    has_detectron2 = False
                if not has_detectron2:
                    policy = os.environ.get("GLOSSAPI_MINERU_MISSING_DETECTRON2", "rapidocr").lower()
                    console.print("[yellow]detectron2 not available for MinerU.[/yellow]")
                    if policy == "stop":
                        console.print("Install detectron2, set GLOSSAPI_MINERU_ALLOW_STUB=0, then retry.")
                        raise typer.Exit(code=1)
                    if policy == "stub":
                        os.environ["GLOSSAPI_MINERU_ALLOW_STUB"] = "1"
                        console.print("[yellow]Proceeding with stub output (set GLOSSAPI_MINERU_ALLOW_STUB=0 to disable).[/yellow]")
                    else:
                        rapidocr_ok = True
                        if importlib.util.find_spec("docling") is None:
                            rapidocr_ok = False
                        else:
                            try:
                                from transformers import AutoModelForImageTextToText  # type: ignore  # noqa: F401
                            except Exception:
                                rapidocr_ok = False
                        if not rapidocr_ok:
                            console.print("[yellow]RapidOCR not available; falling back to stub output.[/yellow]")
                            os.environ["GLOSSAPI_MINERU_ALLOW_STUB"] = "1"
                        else:
                            console.print("[yellow]Switching to RapidOCR (set GLOSSAPI_MINERU_MISSING_DETECTRON2=stub to keep MinerU).[/yellow]")
                            ocr_kwargs["backend"] = "rapidocr"
            # CUDA pre-flight for any backend that needs a GPU.
            backend_name = ocr_kwargs.get("backend", "")
            device_name = ocr_kwargs.get("device", "")
            if device_name == "cuda" and backend_name in _CUDA_BACKEND_PYTHON_ENV:
                _cuda_preflight(backend_name)
            corpus.ocr(**ocr_kwargs)
            return
        corpus.ocr()

    phase_map = {
        "extract": _run_extract,
        "clean": lambda: corpus.clean(**(clean_kwargs or {})),
        "ocr": _run_ocr,
        "section": corpus.section,
        "annotate": corpus.annotate,
        "export-jsonl": lambda: _run_jsonl_export(corpus, output_dir, input_dir),
    }

    for phase in phases:
        if phase not in phase_map:
            console.print(f"[red]Unknown phase:[/red] {phase}")
            raise typer.Exit(code=2)
        if not _maybe_confirm(phase, confirm_each=confirm_each):
            continue
        console.print(Panel.fit(f"Running phase: {phase}", title="GlossAPI"))
        phase_map[phase]()


def _run_mineru_demo() -> None:
    input_dir = Path("samples") / "eellak"
    ocr_dir = input_dir / "ocr"
    text_dir = input_dir / "text"
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"artifacts/mineru_demo_run_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    accel_type = "MPS" if platform.system() == "Darwin" else "CUDA"
    mineru_device = "mps" if accel_type == "MPS" else "cuda"
    mineru_backend = "hybrid-auto-engine" if accel_type == "MPS" else "pipeline"
    corpus = Corpus(input_dir, output_dir)

    ocr_files = [p.name for p in sorted(ocr_dir.glob("*.pdf"))] if ocr_dir.exists() else []
    text_files = [p.name for p in sorted(text_dir.glob("*.pdf"))] if text_dir.exists() else []

    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    def _stage_downloads(src_dir: Path, names: list[str]) -> None:
        for name in names:
            src = src_dir / name
            dst = downloads_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)

    _stage_downloads(ocr_dir, ocr_files)
    _stage_downloads(text_dir, text_files)

    if ocr_files:
        corpus.extract(
            input_format="pdf",
            accel_type=accel_type,
            force_ocr=False,
            phase1_backend="safe",
            filenames=ocr_files,
            file_paths=[ocr_dir / name for name in ocr_files],
            skip_existing=False,
        )

    if text_files:
        corpus.extract(
            input_format="pdf",
            accel_type=accel_type,
            phase1_backend="safe",
            filenames=text_files,
            file_paths=[text_dir / name for name in text_files],
            skip_existing=False,
        )

    corpus.clean(drop_bad=False)
    corpus.ocr(
        backend="mineru",
        math_enhance=True,
        reprocess_completed=True,
        mode="ocr_bad_then_math",
        device=mineru_device,
        mineru_backend=mineru_backend,
    )
    console.print(f"[green]Done! Check results in: {output_dir.resolve()}[/green]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_dir: Optional[str] = typer.Option(None, help="Input directory with PDFs or markdown."),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for pipeline artifacts."),
    input_format: Optional[str] = typer.Option(None, help="Input format: pdf or markdown."),
    phase: Optional[List[str]] = typer.Option(None, "--phase", "-p", help="Phase to run (repeatable)."),
    confirm_each: bool = typer.Option(True, help="Confirm each phase before running."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _run_wizard(
        input_dir=input_dir,
        output_dir=output_dir,
        input_format=input_format,
        phase=phase,
        confirm_each=confirm_each,
        log_level=log_level,
    )


@app.command("wizard")
def wizard() -> None:
    """Interactive wizard (same as default invocation)."""
    _run_wizard(
        input_dir=None,
        output_dir=None,
        input_format=None,
        phase=None,
        confirm_each=True,
        log_level="INFO",
    )


def _run_wizard(
    *,
    input_dir: Optional[str],
    output_dir: Optional[str],
    input_format: Optional[str],
    phase: Optional[List[str]],
    confirm_each: bool,
    log_level: str,
) -> None:
    console.print(Panel.fit("GlossAPI Wizard", title="Welcome"))
    console.print(f"[dim]Detected OS: {_detect_os_label()}[/dim]")
    preset = _ask_preset()

    if preset == "MinerU demo (samples/eellak)":
        _run_mineru_demo()
        return

    if preset == "Lightweight PDF smoke test":
        resolved_format = "pdf"
        resolved_input = Path("samples") / "lightweight_pdf_corpus" / "pdfs"
        resolved_output = Path("artifacts") / "lightweight_pdf_run"
        resolved_phases = ["extract"]
        level = getattr(logging, log_level.upper(), logging.INFO)
        _run_pipeline(
            input_dir=resolved_input,
            output_dir=resolved_output,
            input_format=resolved_format,
            phases=resolved_phases,
            confirm_each=False,
            log_level=level,
        )
        return

    resolved_format = _ask_input_format(input_format)
    resolved_input = _ensure_path(
        input_dir,
        label="Input directory",
        default=_default_input_dir(resolved_format),
        must_exist=True,
    )
    resolved_output = _ensure_path(
        output_dir,
        label="Output directory",
        default=Path("artifacts") / "glossapi_run",
        must_exist=False,
    )
    resolved_phases = _ask_phases(phase)
    resolved_input = _ensure_input_has_files(resolved_input, resolved_format)

    accel_mode = _ask_accel_mode()
    ocr_kwargs = None
    extract_kwargs: dict = {}
    clean_kwargs: dict = {}
    if "ocr" in resolved_phases:
        backend = _ask_ocr_backend()
        if backend == "none":
            resolved_phases = [phase for phase in resolved_phases if phase != "ocr"]
        else:
            ocr_kwargs = {
                "backend": backend,
                "math_enhance": True,
                "mode": "ocr_bad_then_math",
                "fix_bad": True,
            }
            if backend == "mineru":
                device = "mps" if platform.system() == "Darwin" else "cuda"
                mineru_backend = "hybrid-auto-engine" if device == "mps" else "pipeline"
                if accel_mode == "CPU":
                    device = "cpu"
                    mineru_backend = "pipeline"
                ocr_kwargs.update({"device": device, "mineru_backend": mineru_backend})
                extract_kwargs["phase1_backend"] = "safe"
                clean_kwargs["drop_bad"] = False
            if backend == "deepseek-ocr":
                extract_kwargs["phase1_backend"] = "safe"
                clean_kwargs["drop_bad"] = False
                device = "cuda"
                if accel_mode == "CPU":
                    device = "cpu"
                ocr_kwargs.update({"device": device})
            if backend == "deepseek-ocr-2":
                extract_kwargs["phase1_backend"] = "safe"
                clean_kwargs["drop_bad"] = False
                device = "mps" if platform.system() == "Darwin" else "cpu"
                ocr_kwargs.update({"device": device})
            if backend == "olmocr":
                extract_kwargs["phase1_backend"] = "safe"
                clean_kwargs["drop_bad"] = False
                device = "mps" if platform.system() == "Darwin" else "cuda"
                if accel_mode == "CPU":
                    device = "cpu"
                ocr_kwargs.update({"device": device})
            if backend == "glm-ocr":
                extract_kwargs["phase1_backend"] = "safe"
                clean_kwargs["drop_bad"] = False
                device = "mps" if platform.system() == "Darwin" else "cpu"
                ocr_kwargs.update({"device": device})
            if backend == "rapidocr" and accel_mode == "CPU":
                ocr_kwargs.update({"fix_bad": True})

    if resolved_format == "pdf" and (ocr_kwargs is None or ocr_kwargs.get("backend") == "rapidocr"):
        if accel_mode == "CPU":
            extract_kwargs.setdefault("accel_type", "CPU")
        else:
            extract_kwargs.setdefault(
                "accel_type",
                "MPS" if platform.system() == "Darwin" else "CUDA",
            )

    level = getattr(logging, log_level.upper(), logging.INFO)
    _run_pipeline(
        input_dir=resolved_input,
        output_dir=resolved_output,
        input_format=resolved_format,
        phases=resolved_phases,
        confirm_each=confirm_each,
        log_level=level,
        ocr_kwargs=ocr_kwargs,
        extract_kwargs=extract_kwargs or None,
        clean_kwargs=clean_kwargs or None,
    )
