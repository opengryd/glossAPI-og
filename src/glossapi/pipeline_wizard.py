from __future__ import annotations

import logging
import os
import platform
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

from ._cli_utils import (
    RunConfig,
    _gum_choose_impl,
    _gum_confirm_impl,
    _gum_input_impl,
    _run_gum_raw,
)

app = typer.Typer(add_completion=False, help="Interactive wizard for the GlossAPI pipeline.")
console = Console()

PHASE_CHOICES = [
    "extract",
    "clean",
    "ocr",
    "section",
    "annotate",
    "export-jsonl",
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
    return _run_gum_raw(args)


def _gum_choose(
    label: str,
    choices: List[str],
    *,
    default: Optional[List[str] | str] = None,
    multi: bool = False,
) -> List[str]:
    """Thin wrapper that binds the wizard's ``_run_gum`` to the shared impl.

    Always returns a list.  On user cancel (empty single-select) raises
    ``typer.Exit`` so the wizard aborts cleanly.
    """
    result = _gum_choose_impl(_run_gum, label, choices, default=default, multi=multi)
    if not result and not multi:
        raise typer.Exit(code=1)
    return result


def _gum_confirm(label: str, default: bool) -> bool:
    return _gum_confirm_impl(_run_gum, label, default)


def _gum_input(label: str, default: str) -> str:
    result = _gum_input_impl(_run_gum, label, default)
    if result is None:
        raise typer.Exit(code=1)
    return result


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


def _ask_input_format(value: Optional[str], *, detected_default: Optional[str] = None) -> str:
    if value:
        return _normalize_input_format(value)
    # Use the detected value as the pre-selected default but always show the
    # menu so the user can confirm or override.
    default = detected_default if detected_default in ("pdf", "md") else "pdf"
    default_label = "markdown" if default == "md" else default
    if detected_default:
        console.print(f"[dim]  (detected from directory contents — change if needed)[/dim]")
    choice = _gum_choose("Input format", ["pdf", "markdown"], default=default_label)
    return _normalize_input_format(choice[0])


def _ask_force_cpu() -> bool:
    """Ask whether to force CPU-only mode.  Returns ``True`` when opted in."""
    return _gum_confirm("Force CPU-only mode? (skips GPU acceleration)", default=False)


def _infer_format_from_dir(d: Path) -> Optional[str]:
    """Return ``'pdf'`` or ``'md'`` when the directory contains only one type.

    Returns ``None`` when the directory is empty or has both types (ambiguous).
    """
    has_pdf = any(d.glob("*.pdf"))
    has_md = any(d.glob("*.md")) or any(d.glob("*.markdown"))
    if has_pdf and not has_md:
        return "pdf"
    if has_md and not has_pdf:
        return "md"
    return None


def _last_output_dir() -> Path:
    """Return the output directory used in the last pipeline run (or the default)."""
    try:
        from . import _state
        last = _state.get_pipeline_state("last_output_dir")
        if last:
            return Path(last)
    except Exception:
        pass
    return Path("artifacts") / "glossapi_run"


def _make_run_output_dir(backend: Optional[str] = None) -> Path:
    """Generate a fresh timestamped output directory path.

    Returns ``artifacts/run_<backend-slug>_<YYYYMMDD-HHmm>`` so every
    pipeline run lands in a uniquely-named, self-describing folder without
    any manual renaming.  When no OCR backend is selected the slug is
    ``noocr``.
    """
    from datetime import datetime
    slug = backend if (backend and backend != "none") else "noocr"
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return Path("artifacts") / f"run_{slug}_{ts}"


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


# ---------------------------------------------------------------------------
# Backend → env-var auto-injection
# ---------------------------------------------------------------------------

# Maps each subprocess-based backend to the (GLOSSAPI_* env var, relative path
# within its venv) that its runner consults at runtime.  Backends that are
# imported in-process (rapidocr) are NOT listed here — they need the *active*
# venv to contain their deps, which is a separate concern.
_BACKEND_ENV_MAP: dict[str, tuple[str, str]] = {
    "mineru":         ("GLOSSAPI_MINERU_COMMAND",           "bin/magic-pdf"),
    "deepseek-ocr":   ("GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON", "bin/python"),
    "deepseek-ocr-2": ("GLOSSAPI_DEEPSEEK2_PYTHON",         "bin/python"),
    "glm-ocr":        ("GLOSSAPI_GLMOCR_PYTHON",            "bin/python"),
    "olmocr":         ("GLOSSAPI_OLMOCR_PYTHON",            "bin/python"),
}


def _inject_backend_env(backend: str) -> None:
    """Auto-set GLOSSAPI_* env vars so the selected backend can find its venv.

    This is the fix for the class of failures where ``glossapi setup`` was run
    for one profile (e.g. *deepseek-ocr*) but ``glossapi pipeline`` selected a
    different backend (e.g. *mineru*) — the second backend's CLI tool or Python
    binary was not on PATH because its venv was never activated in the shell.

    Resolution strategy
    -------------------
    1. Look up the backend's venv via the state file (written by ``glossapi
       setup``, most precise).
    2. Fall back to a filesystem scan of ``dependency_setup/.venvs/``.
    3. Set the appropriate ``GLOSSAPI_*`` env var to the absolute path of the
       binary/executable inside that venv.
    4. Any var already set in the environment is left untouched — explicit user
       overrides always win.
    """
    if backend not in _BACKEND_ENV_MAP:
        return

    env_var, rel_bin = _BACKEND_ENV_MAP[backend]

    # Respect explicit user override.
    if os.environ.get(env_var):
        return

    venv = _find_venv_for_backend(backend)
    if venv is None:
        return  # No venv found; the runner will emit its own actionable error.

    bin_path = venv / rel_bin
    if bin_path.exists():
        os.environ[env_var] = str(bin_path)
        console.print(f"[dim]  auto-configured {env_var} → {bin_path}[/dim]")
    else:
        console.print(
            f"[yellow]  {backend}: venv found at {venv} but '{rel_bin}' is missing.\n"
            f"  Re-run [bold]glossapi setup --mode {backend}[/bold] to rebuild the environment.[/yellow]"
        )


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


def _ask_ocr_backend(
    *,
    default: Optional[str] = None,
    readiness: Optional[dict] = None,
) -> str:
    """Prompt for OCR backend, annotating each option with its readiness.

    Parameters
    ----------
    default:
        Pre-select this backend (e.g. the backend just installed by
        ``glossapi setup``).  Falls back to the platform default when ``None``.
    readiness:
        Pre-computed ``_backend_readiness()`` dict.  Recomputed internally when
        ``None`` so callers that already have it can avoid a second lookup.
    """
    if readiness is None:
        readiness = _backend_readiness()
    default_backend = default or ("mineru" if platform.system() == "Darwin" else "rapidocr")

    has_any_info = bool(readiness)

    # When exactly one backend is ready, pre-select it in the menu but still
    # show all options — the user sees why it is pre-selected (✓) and can still
    # choose something else (e.g. to run with stub output).
    ready_backends = [b for b in _ALL_OCR_BACKENDS if b != "none" and readiness.get(b) is True]
    auto_preselect = ready_backends[0] if len(ready_backends) == 1 else None

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

    # Prefer in order: (1) auto-preselect (only ready backend), (2) first ready
    # platform-default, (3) platform-default regardless of readiness.
    default_label: str
    if auto_preselect:
        default_label = next(
            (lbl for lbl in labels if label_to_backend[lbl] == auto_preselect),
            labels[0],
        )
        console.print(
            f"[dim]  {auto_preselect} is pre-selected because it is the only ready backend[/dim]"
        )
    else:
        ready_default = next(
            (lbl for lbl in labels if label_to_backend[lbl] == default_backend and readiness.get(default_backend) is True),
            None,
        )
        if ready_default:
            default_label = ready_default
        else:
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


def _preflight_backend_or_warn(backend: str, readiness: dict[str, bool | None]) -> None:
    """Block (or offer a stub escape hatch) when the selected backend is not installed.

    If ``readiness[backend]`` is definitively ``False``:

    * Print a clear error panel with the exact setup command to run.
    * Offer a stub-output escape hatch so the user can still walk through the
      pipeline structure without real OCR (useful for CI, demos, dry-runs).
    * If the user declines the stub hatch, exit with code 1.

    When readiness is ``True`` or ``None`` (unknown — no state file), this is
    a no-op so the pipeline always works on machines that have never run
    ``glossapi setup`` but have the deps installed in the active environment.
    """
    if readiness.get(backend) is not False:
        return

    console.print(
        f"\n[bold red]✗  Backend '{backend}' is not set up.[/bold red]\n"
        f"   No installation was found in the state file or on disk.\n"
        f"   Run:  [bold]glossapi setup --mode {backend}[/bold]  first.\n"
    )
    stub_env_var = f"GLOSSAPI_{backend.upper().replace('-', '_')}_ENABLE_STUB"
    if _gum_confirm(
        f"Proceed with placeholder (stub) output for '{backend}'?"
        " (no real OCR — useful only for testing pipeline flow)",
        default=False,
    ):
        os.environ[stub_env_var] = "1"
        console.print(
            f"[yellow]  {stub_env_var}=1 — stub output active.\n"
            f"  Run [bold]glossapi setup --mode {backend}[/bold] to enable real OCR.[/yellow]"
        )
    else:
        raise typer.Exit(code=1)


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


def _run_pipeline(cfg: RunConfig) -> None:
    """Execute the pipeline phases described by *cfg*.

    Accepts a fully-populated ``RunConfig`` produced by ``_run_wizard``.
    All per-phase configuration (backend, device, extract options, clean
    options) is read from the typed fields of *cfg*; there are no opaque
    ``*_kwargs`` dicts to track.
    """
    corpus = Corpus(input_dir=cfg.input_dir, output_dir=cfg.output_dir, log_level=cfg.log_level)

    def _run_extract() -> None:
        if cfg.input_format == "md":
            markdown_dir = cfg.output_dir / "markdown"
            downloads_dir = cfg.output_dir / "downloads"
            markdown_dir.mkdir(parents=True, exist_ok=True)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            candidates = list(cfg.input_dir.glob("*.md")) + list(cfg.input_dir.glob("*.markdown"))
            if not candidates:
                console.print(f"[yellow]No markdown files found in {cfg.input_dir}.[/yellow]")
                return
            for path in candidates:
                shutil.copy2(path, markdown_dir / path.name)
                shutil.copy2(path, downloads_dir / path.name)
            console.print(f"[green]Copied {len(candidates)} markdown files into {markdown_dir}.[/green]")
            return
        if cfg.phase1_backend == "safe":
            previous = os.environ.get("GLOSSAPI_SKIP_DOCLING_BOOT")
            os.environ["GLOSSAPI_SKIP_DOCLING_BOOT"] = "1"
            try:
                corpus.extract(input_format=cfg.input_format, **(cfg.extract_kwargs or {}))
            finally:
                if previous is None:
                    os.environ.pop("GLOSSAPI_SKIP_DOCLING_BOOT", None)
                else:
                    os.environ["GLOSSAPI_SKIP_DOCLING_BOOT"] = previous
            return
        if importlib.util.find_spec("docling") is None:
            console.print("[red]Missing dependency: docling.[/red]")
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
        corpus.extract(input_format=cfg.input_format, **(cfg.extract_kwargs or {}))

    def _run_ocr() -> None:
        if cfg.backend is not None:
            if cfg.backend == "mineru":
                env_cmd = os.environ.get("GLOSSAPI_MINERU_COMMAND")
                candidate = Path(env_cmd).expanduser() if env_cmd else None
                cmd_exists = candidate.exists() if candidate else False
                if not cmd_exists and shutil.which("magic-pdf") is None:
                    console.print("[red]Missing MinerU CLI (magic-pdf).[/red]")
                    console.print("Run glossapi setup and choose the MinerU profile to install dependencies.")
                    console.print("Set GLOSSAPI_MINERU_COMMAND or install magic-pdf, then retry.")
                    raise typer.Exit(code=1)
                # Binary confirmed — enable real OCR.  magic-pdf runs as a subprocess
                # in its own venv, so detectron2 does NOT need to be importable in this
                # (host) process.  Respect any explicit user override.
                os.environ.setdefault("GLOSSAPI_MINERU_ENABLE_OCR", "1")
                try:
                    import detectron2  # type: ignore  # noqa: F401
                    has_detectron2 = True
                except Exception:
                    has_detectron2 = False
                if not has_detectron2:
                    policy = os.environ.get("GLOSSAPI_MINERU_MISSING_DETECTRON2", "rapidocr").lower()
                    # detectron2 is absent from the *host* env; this is non-fatal when
                    # the magic-pdf CLI is ready (it manages its own environment).
                    console.print("[yellow]detectron2 not available in host env (non-fatal for MinerU CLI runner).[/yellow]")
                    if policy == "stop":
                        console.print("Install detectron2 in the host env, or set GLOSSAPI_MINERU_MISSING_DETECTRON2=stub to continue with stub output.")
                        raise typer.Exit(code=1)
                    if policy == "stub":
                        os.environ["GLOSSAPI_MINERU_ENABLE_OCR"] = "0"
                        os.environ["GLOSSAPI_MINERU_ENABLE_STUB"] = "1"
                        console.print("[yellow]Proceeding with stub output (set GLOSSAPI_MINERU_ENABLE_STUB=0 to disable).[/yellow]")
                    else:
                        # Default ("rapidocr"): switch only when rapidocr is genuinely
                        # available in the current env; otherwise, stay with the MinerU
                        # CLI which is already confirmed ready.
                        rapidocr_ok = True
                        if importlib.util.find_spec("docling") is None:
                            rapidocr_ok = False
                        else:
                            try:
                                from transformers import AutoModelForImageTextToText  # type: ignore  # noqa: F401
                            except Exception:
                                rapidocr_ok = False
                        if rapidocr_ok:
                            console.print("[yellow]Switching to RapidOCR (set GLOSSAPI_MINERU_MISSING_DETECTRON2=stub to keep MinerU).[/yellow]")
                            cfg.backend = "rapidocr"  # RunConfig is mutable; ocr_kwargs property reflects this
                        # else: MinerU CLI is confirmed ready; proceed without detectron2 in host env.
            # CUDA pre-flight for any backend that needs a GPU.
            if cfg.device == "cuda" and cfg.backend in _CUDA_BACKEND_PYTHON_ENV:
                _cuda_preflight(cfg.backend)  # type: ignore[arg-type]
            corpus.ocr(**(cfg.ocr_kwargs or {}))
            return
        corpus.ocr()

    phase_map = {
        "extract": _run_extract,
        "clean": lambda: corpus.clean(**cfg.clean_kwargs),
        "ocr": _run_ocr,
        "section": corpus.section,
        "annotate": corpus.annotate,
        "export-jsonl": lambda: _run_jsonl_export(corpus, cfg.output_dir, cfg.input_dir),
    }

    for phase in cfg.phases:
        if phase not in phase_map:
            console.print(f"[red]Unknown phase:[/red] {phase}")
            raise typer.Exit(code=2)
        if not _maybe_confirm(phase, confirm_each=cfg.confirm_each):
            continue
        console.print(Panel.fit(f"Running phase: {phase}", title="GlossAPI"))
        phase_map[phase]()

    # Emit a single consolidated Performance & Power Report at the end of the run.
    # Pass cfg.backend (the user-selected OCR backend, e.g. "mineru") so the
    # report is labelled correctly even when the OCR phase was skipped and no
    # OCR perf sample was recorded.
    try:
        corpus.perf_report(backend=cfg.backend or None)
    except Exception:
        pass


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_dir: Optional[str] = typer.Option(None, help="Input directory with PDFs or markdown."),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for pipeline artifacts."),
    input_format: Optional[str] = typer.Option(None, help="Input format: pdf or markdown."),
    phase: Optional[List[str]] = typer.Option(None, "--phase", "-p", help="Phase to run (repeatable)."),
    confirm_each: bool = typer.Option(False, help="Confirm each phase before running."),
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
        confirm_each=False,
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
    default_backend: Optional[str] = None,
) -> None:
    console.print(Panel.fit("GlossAPI Wizard", title="Welcome"))

    # 1. Input directory — ask first so we can infer format from its contents.
    resolved_input = _ensure_path(
        input_dir,
        label="Input directory",
        default=Path("samples") / "lightweight_pdf_corpus" / "pdfs",
        must_exist=True,
    )

    # 2. Input format — probe the directory to set a smart default, then always
    #    ask so the user confirms or overrides.
    if input_format:
        resolved_format = _normalize_input_format(input_format)
    else:
        inferred = _infer_format_from_dir(resolved_input)
        resolved_format = _ask_input_format(None, detected_default=inferred)

    # 3. Verify files exist (retry loop built into helper).
    resolved_input = _ensure_input_has_files(resolved_input, resolved_format)

    # 4. Phases — ask early so we know whether OCR is needed before prompting
    #    for the output directory.
    resolved_phases = _ask_phases(phase)

    # 5. CPU override — simple yes/no instead of a vague CPU/GPU menu.
    force_cpu = _ask_force_cpu()

    # 6. OCR backend — resolve before asking for the output directory so that
    #    the backend slug can be embedded in the default folder name.
    _readiness: dict = {}
    resolved_backend: Optional[str] = None
    if "ocr" in resolved_phases:
        _readiness = _backend_readiness()
        _chosen = _ask_ocr_backend(default=default_backend, readiness=_readiness)
        if _chosen != "none":
            resolved_backend = _chosen

    # 7. Output directory — default is a fresh timestamped path that embeds
    #    the backend slug: artifacts/run_<backend>_<YYYYMMDD-HHmm>.
    resolved_output = _ensure_path(
        output_dir,
        label="Output directory",
        default=_make_run_output_dir(resolved_backend),
        must_exist=False,
    )

    # Compute log level early so it can be embedded in RunConfig.
    level = getattr(logging, log_level.upper(), logging.INFO)

    # -------------------------------------------------------------------------
    # Build a typed RunConfig from all wizard answers.
    # -------------------------------------------------------------------------
    cfg = RunConfig(
        input_dir=resolved_input,
        output_dir=resolved_output,
        input_format=resolved_format,
        phases=resolved_phases,
        confirm_each=confirm_each,
        log_level=level,
    )

    if "ocr" in resolved_phases and resolved_backend is None:
        # User chose "none" backend — strip the OCR phase.
        cfg.phases = [p for p in cfg.phases if p != "ocr"]
    elif resolved_backend is not None:
        _preflight_backend_or_warn(resolved_backend, _readiness)
        _inject_backend_env(resolved_backend)
        cfg.backend = resolved_backend
        if resolved_backend == "mineru":
            cfg.device = "mps" if platform.system() == "Darwin" else "cuda"
            cfg.mineru_backend = "hybrid-auto-engine" if cfg.device == "mps" else "pipeline"
            if force_cpu:
                cfg.device = "cpu"
                cfg.mineru_backend = "pipeline"
            cfg.phase1_backend = "safe"
            cfg.drop_bad = False
        if resolved_backend == "deepseek-ocr":
            cfg.phase1_backend = "safe"
            cfg.drop_bad = False
            if force_cpu:
                cfg.device = "cpu"
            elif platform.system() == "Darwin":
                cfg.device = "mps"
            else:
                cfg.device = "cuda"
        if resolved_backend == "deepseek-ocr-2":
            cfg.phase1_backend = "safe"
            cfg.drop_bad = False
            cfg.device = "mps" if platform.system() == "Darwin" else "cpu"
        if resolved_backend == "olmocr":
            cfg.phase1_backend = "safe"
            cfg.drop_bad = False
            cfg.device = "mps" if platform.system() == "Darwin" else "cuda"
            if force_cpu:
                cfg.device = "cpu"
        if resolved_backend == "glm-ocr":
            cfg.phase1_backend = "safe"
            cfg.drop_bad = False
            cfg.device = "mps" if platform.system() == "Darwin" else "cpu"
        if resolved_backend == "rapidocr" and force_cpu:
            cfg.device = "cpu"

    if cfg.input_format == "pdf" and (cfg.backend is None or cfg.backend == "rapidocr"):
        cfg.accel_type = "CPU" if force_cpu else (
            "MPS" if platform.system() == "Darwin" else "CUDA"
        )

    # Persist the chosen output directory for next time.
    try:
        from . import _state
        _state.set_pipeline_state("last_output_dir", str(cfg.output_dir))
    except Exception:
        pass

    # Power monitoring: prompt for sudo password if powermetrics needs it,
    # then prewarm the sensor so it's cached before the pipeline run starts.
    try:
        from .perf_metrics import (
            needs_powermetrics_sudo as _needs_sudo,
            set_powermetrics_sudo_password as _set_pwd,
            prewarm_power_sensor as _prewarm,
        )
        if _needs_sudo():
            console.print("")
            console.print(
                "[bold]Power monitoring[/bold] — [dim]powermetrics measures CPU+GPU+ANE energy "
                "(Apple Silicon). It requires sudo access.[/dim]"
            )
            console.print("[dim]Leave blank to skip and report timing only.[/dim]")
            try:
                pwd = typer.prompt("  sudo password", default="", hide_input=True)
            except Exception:
                pwd = ""
            if pwd:
                ok = _set_pwd(pwd)
                if not ok:
                    console.print(
                        "[yellow]sudo password rejected — power monitoring unavailable.[/yellow]"
                    )
        sensor_name = _prewarm()
        if sensor_name is not None:
            console.print(f"[dim]Power monitoring: {sensor_name}[/dim]")
        else:
            console.print("[dim]Power monitoring: unavailable (timing only)[/dim]")
    except Exception:
        pass

    _run_pipeline(cfg)
