from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
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


def _detect_os_label() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macOS Apple Silicon (MPS/Metal + MLX supported)"
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


_ALL_OCR_BACKENDS = ["none", "rapidocr", "mineru", "deepseek-ocr", "deepseek-ocr-2", "glm-ocr", "olmocr"]


def _find_venv_for_backend(mode: str) -> Optional[Path]:
    """Return the venv Path for *mode* if a valid one exists on disk, else None.

    The setup script creates venvs at::

        dependency_setup/.venvs/{mode}[-py{tag}]/

    e.g. ``deepseek-ocr-py312/``, ``rapidocr/``, ``mineru-py311/``.
    We glob for any directory whose name starts with ``{mode}`` and contains a
    working ``bin/python`` executable, preferring an exact match over tagged ones.
    """
    # Try the path recorded in the state file first (most precise).
    try:
        from . import _state as _st
        recorded = _st.get_backend_venv(mode)
        if recorded and (recorded / "bin" / "python").exists():
            return recorded
    except Exception:
        pass

    # Fall back to scanning the default .venvs directory.
    venvs_root = Path("dependency_setup") / ".venvs"
    if not venvs_root.is_dir():
        return None

    candidates = sorted(venvs_root.iterdir())
    # Exact match first, then prefix-tagged variants (e.g. deepseek-ocr-py312).
    for d in candidates:
        if d.name == mode and (d / "bin" / "python").exists():
            return d
    for d in candidates:
        if d.name.startswith(f"{mode}-py") and (d / "bin" / "python").exists():
            return d
    return None


def _backend_readiness() -> dict[str, bool | None]:
    """Return a {backend: ready?} mapping.

    Priority order per backend:
    1. State file entry (set by ``glossapi setup``) — authoritative.
    2. Filesystem probe: does ``dependency_setup/.venvs/{mode}*/bin/python`` exist?
    3. Last-resort heuristic only when *neither* a state file *nor* a discoverable
       venv exists (covers vanilla installs where everything shares one env).

    Values:
    ``True``  — definitely installed/ready.
    ``False`` — definitely not set up.
    ``None``  — genuinely unknown (no evidence either way).
    """
    try:
        from . import _state
        record = _state.get_installed_backends()
        has_file = _state.has_state_file()
    except Exception:
        record = {}
        has_file = False

    result: dict[str, bool | None] = {}

    # 1. State file is authoritative for backends it records.
    for b, info in record.items():
        result[b] = bool(info.get("installed"))

    # 2. For backends not yet in the state file, check the venvs directory.
    #    If the state file exists but a backend has no entry there AND no venv
    #    is found on disk → it's not installed.
    for b in _ALL_OCR_BACKENDS:
        if b == "none" or b in result:
            continue
        venv = _find_venv_for_backend(b)
        if venv is not None:
            result[b] = True
        elif has_file:
            # State file exists; this backend was never set up.
            result[b] = False
        # else: no state file and no venv found → leave as None (unknown).

    # 3. Last-resort import/CLI heuristic: only when there is no state file
    #    AND no backend venv was found anywhere on disk.  If the .venvs dir
    #    has any backend entry it means the setup script has been used before,
    #    so presence of docling in the current env is NOT evidence that
    #    rapidocr was set up (it could be a transitive dep of another backend).
    any_venv_found = any(v is True for v in result.values())
    if not has_file and not any_venv_found:
        if "rapidocr" not in result:
            result["rapidocr"] = importlib.util.find_spec("docling") is not None

        if "mineru" not in result:
            env_cmd = os.environ.get("GLOSSAPI_MINERU_COMMAND")
            candidate = Path(env_cmd).expanduser() if env_cmd else None
            if (candidate and candidate.exists()) or shutil.which("magic-pdf"):
                result["mineru"] = True
    elif any_venv_found:
        # The setup script has been used — any backend we couldn't find a venv
        # for (and that the state file doesn't mention) is definitively not installed.
        for b in _ALL_OCR_BACKENDS:
            if b != "none" and b not in result:
                result[b] = False

    return result


def _ask_ocr_backend() -> str:
    """Prompt for OCR backend, annotating each option with its readiness."""
    default_backend = "mineru" if platform.system() == "Darwin" else "rapidocr"
    readiness = _backend_readiness()
    has_any_info = bool(readiness)

    # Build annotated labels and a reverse map label → bare backend name.
    labels: list[str] = []
    label_to_backend: dict[str, str] = {}
    for b in _ALL_OCR_BACKENDS:
        if not has_any_info or b == "none":
            label = b
        elif readiness.get(b) is True:
            label = f"✓  {b}"
        elif readiness.get(b) is False:
            label = f"✗  {b}  (not set up — run: glossapi setup --mode {b})"
        else:
            label = b  # unknown
        labels.append(label)
        label_to_backend[label] = b

    # Prefer the first ready backend as default, fall back to platform default.
    default_label: str
    ready_default = next(
        (lbl for lbl in labels if label_to_backend[lbl] == default_backend and readiness.get(default_backend) is True),
        None,
    )
    if ready_default:
        default_label = ready_default
    else:
        # Fall back: whichever label corresponds to the platform default.
        default_label = next(
            (lbl for lbl in labels if label_to_backend[lbl] == default_backend),
            labels[0],
        )

    if has_any_info:
        n_ready = sum(1 for b in _ALL_OCR_BACKENDS if b != "none" and readiness.get(b) is True)
        if n_ready == 0:
            console.print(
                "[yellow]No OCR backends are set up yet.[/yellow] "
                "Run [bold]glossapi setup[/bold] first, or choose a backend to run with stub output."
            )
        else:
            console.print(f"[dim]{n_ready} backend(s) ready (✓). Others need 'glossapi setup'.[/dim]")

    choice = _gum_choose("OCR backend", labels, default=default_label)
    raw_label = choice[0]
    return label_to_backend.get(raw_label, raw_label.split()[0].lstrip("✓✗ "))


def _maybe_confirm(phase: str, *, confirm_each: bool) -> bool:
    if not confirm_each:
        return True
    return _gum_confirm(f"Run phase: {phase}?", default=True)


# Backend names that may run on CUDA, mapped to their configured Python binary.
# The wizard probes the binary for ``torch.cuda.is_available()`` only when
# ``device='cuda'`` is explicitly set for the backend (see _run_wizard logic).
# OlmOCR is dual-platform: CUDA/vLLM on Linux, MLX/MPS on macOS. On macOS the
# wizard sets device='mps' so CUDA probing is never triggered for OlmOCR there.
# DeepSeek OCR v2 and GLM-OCR are MLX-only (macOS); they are never in this map.
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
    """Pre-flight check: verify CUDA is available when a backend is set to run on CUDA.

    Only called when ``device='cuda'`` is explicitly selected by the wizard
    (e.g. OlmOCR on Linux, DeepSeek-OCR on Linux).  On macOS, OlmOCR uses
    MLX/MPS and DeepSeek OCR v2 / GLM-OCR use MLX — the wizard sets
    ``device='mps'`` or ``device='cpu'`` for those combinations and this
    function is never reached.

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
    stub_env = f"GLOSSAPI_{backend.upper().replace('-', '_')}_ENABLE_STUB"

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
                        console.print("Install detectron2, set GLOSSAPI_MINERU_ENABLE_STUB=0, then retry.")
                        raise typer.Exit(code=1)
                    if policy == "stub":
                        os.environ["GLOSSAPI_MINERU_ENABLE_STUB"] = "1"
                        console.print("[yellow]Proceeding with stub output (set GLOSSAPI_MINERU_ENABLE_STUB=0 to disable).[/yellow]")
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
                            os.environ["GLOSSAPI_MINERU_ENABLE_STUB"] = "1"
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

    # Emit a single consolidated Performance & Power Report at the end of the run.
    try:
        corpus.perf_report()
    except Exception:
        pass


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
                if accel_mode == "CPU":
                    device = "cpu"
                elif platform.system() == "Darwin":
                    device = "mps"
                else:
                    device = "cuda"
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
                ocr_kwargs.update({"fix_bad": True, "device": "cpu"})

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
