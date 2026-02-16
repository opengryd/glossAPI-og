"""OlmOCR-2 runner with in-process MLX, CLI dispatch, and stub fallback.

OlmOCR-2 (https://github.com/allenai/olmocr) is a VLM-based OCR toolkit that
converts PDFs into clean Markdown.  It uses a fine-tuned Qwen2.5-VL model and
supports both CUDA (via vLLM) and Apple Silicon MPS (via MLX).

The runner tries the following strategies in order:

1. **In-process MLX** (``allow_inproc=True``, default on macOS when ``mlx_vlm``
   is importable):
   Load the model once via :pymod:`glossapi.ocr.olmocr.mlx_cli` and process
   every PDF without spawning a subprocess.  This is the fast path on Apple
   Silicon — the model stays loaded in memory across files.

2. **MLX CLI subprocess** (``allow_mlx_cli=True``):
   Shell out to ``python -m glossapi.ocr.olmocr.mlx_cli`` (or a user-specified
   script via ``GLOSSAPI_OLMOCR_MLX_SCRIPT``).  Useful when the main venv lacks
   ``mlx-vlm`` but a separate venv does.

3. **OlmOCR CLI subprocess** (``allow_cli=True``):
   Shell out to ``python -m olmocr.pipeline <workspace> --markdown --pdfs ...``.
   The OlmOCR pipeline manages its own vLLM instance, renders PDF pages via
   poppler, and writes Markdown output.  Requires CUDA GPU.

4. **Stub** (``allow_stub=True``):
   Emit placeholder markdown + metrics.  Useful for dry-runs and testing.

OlmOCR inlines equations — Phase-2 math enrichment is not required.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"

# Resolve the embedded MLX CLI script shipped with the package.
_PACKAGE_MLX_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
    except Exception:
        return 0


def _candidate_input_roots(input_root: Path, output_root: Path) -> List[Path]:
    """Return directories to search for input PDFs, in priority order."""
    return [input_root, output_root / "downloads", input_root / "downloads"]


def _resolve_pdf_paths(
    file_list: Sequence[str],
    candidate_roots: Sequence[Path],
) -> Tuple[List[Path], List[str]]:
    """Resolve each filename to an absolute path, searching candidate roots."""
    resolved: List[Path] = []
    missing: List[str] = []
    for name in file_list:
        raw = Path(name)
        if raw.is_absolute():
            resolved.append(raw)
            if not raw.exists():
                missing.append(name)
            continue
        found: Optional[Path] = None
        for root in candidate_roots:
            candidate = (root / name).resolve()
            if candidate.exists():
                found = candidate
                break
        if found is None:
            candidate = raw.resolve()
            if candidate.exists():
                found = candidate
        if found is None:
            missing.append(name)
            found = (candidate_roots[0] / name).resolve()
        resolved.append(found)
    return resolved, missing


# ---------------------------------------------------------------------------
# Strategy 1: In-process MLX execution (macOS Apple Silicon)
# ---------------------------------------------------------------------------


def _can_import_mlx() -> bool:
    """Return True if mlx-vlm is importable in the current process."""
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


def _run_inproc(
    resolved_paths: List[Path],
    file_list: List[str],
    out_root: Path,
    md_dir: Path,
    metrics_dir: Path,
    *,
    model_dir: Optional[Path],
    max_pages: Optional[int],
    content_debug: bool,
) -> Dict[str, Any]:
    """Process PDFs in-process using the embedded mlx_cli module."""
    from . import mlx_cli

    model_path = mlx_cli.resolve_model_dir(str(model_dir) if model_dir else None)
    LOGGER.info("Loading OlmOCR-2 MLX model from %s", model_path)
    model, processor = mlx_cli.load_model_and_processor(model_path)

    results: Dict[str, Any] = {}
    total_files = len(file_list)
    for file_idx, (name, pdf_path) in enumerate(zip(file_list, resolved_paths), 1):
        stem = Path(name).stem
        if not pdf_path.exists():
            LOGGER.warning("OlmOCR MLX: PDF not found: %s", pdf_path)
            results[stem] = {"page_count": 0}
            continue
        LOGGER.info(
            "OlmOCR MLX: processing file %d/%d — %s",
            file_idx, total_files, pdf_path.name,
        )
        try:
            page_count = mlx_cli.process_pdf(
                pdf_path,
                out_root,
                model,
                processor,
                max_pages=max_pages,
                content_debug=content_debug,
            )
            results[stem] = {"page_count": page_count}
        except Exception as exc:
            LOGGER.error(
                "OlmOCR MLX in-process failed for %s: %s", pdf_path.name, exc,
            )
            raise
    return results


# ---------------------------------------------------------------------------
# Strategy 2: MLX CLI subprocess (macOS, separate venv)
# ---------------------------------------------------------------------------


def _resolve_mlx_cli_script() -> Path:
    """Return the path to the MLX CLI script to invoke as a subprocess.

    Priority: ``GLOSSAPI_OLMOCR_MLX_SCRIPT`` env var > package-embedded script.
    """
    env_script = os.environ.get("GLOSSAPI_OLMOCR_MLX_SCRIPT", "").strip()
    if env_script:
        p = Path(env_script)
        if p.exists():
            return p
        LOGGER.warning(
            "GLOSSAPI_OLMOCR_MLX_SCRIPT=%s does not exist; using package script",
            env_script,
        )
    return _PACKAGE_MLX_CLI_SCRIPT


def _pick_cli_input_root(
    file_list: Sequence[str],
    resolved_paths: Sequence[Path],
    candidate_roots: Sequence[Path],
) -> Path:
    """Pick the best input root directory for the CLI subprocess."""
    parents = {path.parent for path in resolved_paths if path.exists()}
    if len(parents) == 1:
        return next(iter(parents))
    best_root = candidate_roots[0]
    best_hits = -1
    extra_roots = list(candidate_roots) + sorted(parents)
    for root in extra_roots:
        hits = sum(1 for name in file_list if (root / Path(name).name).exists())
        if hits > best_hits:
            best_hits = hits
            best_root = root
    return best_root


def _run_mlx_cli(
    input_dir: Path,
    output_dir: Path,
    *,
    python_bin: Optional[Path],
    script: Path,
    model_dir: Optional[Path],
    max_pages: Optional[int],
    content_debug: bool,
    device: Optional[str],
) -> None:
    """Invoke the OlmOCR MLX CLI as a subprocess."""
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)
    cmd: List[str] = [
        str(python_exe),
        str(script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if model_dir is not None:
        cmd += ["--model-dir", str(model_dir)]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if device:
        cmd += ["--device", str(device)]

    env = os.environ.copy()
    LOGGER.info("Running OlmOCR MLX CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


# ---------------------------------------------------------------------------
# Strategy 3: OlmOCR CLI helpers (CUDA / vLLM)
# ---------------------------------------------------------------------------


def _build_cli_cmd(
    *,
    python_bin: Path,
    workspace: Path,
    pdf_paths: List[Path],
    model: str,
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    target_longest_image_dim: Optional[int] = None,
    workers: Optional[int] = None,
    pages_per_group: Optional[int] = None,
) -> List[str]:
    """Build the ``python -m olmocr.pipeline`` command list."""
    cmd: List[str] = [
        str(python_bin),
        "-m",
        "olmocr.pipeline",
        str(workspace),
        "--markdown",
    ]
    if model:
        cmd += ["--model", model]
    if server:
        cmd += ["--server", server]
    if api_key:
        cmd += ["--api_key", api_key]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len is not None:
        cmd += ["--max_model_len", str(max_model_len)]
    if tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if target_longest_image_dim is not None:
        cmd += ["--target_longest_image_dim", str(target_longest_image_dim)]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if pages_per_group is not None:
        cmd += ["--pages_per_group", str(pages_per_group)]
    # Append PDF paths at the end
    cmd.append("--pdfs")
    cmd += [str(p) for p in pdf_paths]
    return cmd


def _run_cli(
    *,
    python_bin: Path,
    workspace: Path,
    pdf_paths: List[Path],
    model: str,
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    target_longest_image_dim: Optional[int] = None,
    workers: Optional[int] = None,
    pages_per_group: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Invoke the OlmOCR pipeline as a subprocess."""
    cmd = _build_cli_cmd(
        python_bin=python_bin,
        workspace=workspace,
        pdf_paths=pdf_paths,
        model=model,
        server=server,
        api_key=api_key,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        target_longest_image_dim=target_longest_image_dim,
        workers=workers,
        pages_per_group=pages_per_group,
    )
    LOGGER.info("Running OlmOCR CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env or os.environ.copy())  # nosec: controlled arguments


def _collect_cli_results(
    file_list: List[str],
    resolved_paths: List[Path],
    workspace_md_dir: Path,
    target_md_dir: Path,
    metrics_dir: Path,
) -> Dict[str, Any]:
    """Collect markdown output from the OlmOCR workspace and copy to target dirs.

    OlmOCR writes markdown to ``<workspace>/markdown/``.  This function copies
    them to the glossAPI-standard ``<output_dir>/markdown/`` location and
    produces metrics JSON sidecars.
    """
    results: Dict[str, Any] = {}
    for name, pdf_path in zip(file_list, resolved_paths):
        stem = Path(name).stem
        md_target = target_md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"

        # OlmOCR preserves input basename; check both stem.md and name.md
        md_src = workspace_md_dir / f"{stem}.md"
        if not md_src.exists():
            # Try with the full filename stem (e.g. "doc.pdf" → "doc.pdf.md"?)
            md_src = workspace_md_dir / f"{Path(name).name}.md"

        if md_src.exists() and md_src != md_target:
            md_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(md_src, md_target)
        elif not md_target.exists():
            # Produce a placeholder if OlmOCR didn't generate output
            md_target.parent.mkdir(parents=True, exist_ok=True)
            md_target.write_text(
                f"# OlmOCR — {pdf_path.name}\n\n[[Blank page]]\n",
                encoding="utf-8",
            )

        page_count = _page_count(pdf_path)
        metrics = {"page_count": page_count}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        results[stem] = metrics

    return results


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


def _write_stub(
    pdf_path: Path,
    md_out: Path,
    metrics_out: Path,
    content_debug: bool,
) -> Dict[str, Any]:
    """Produce placeholder markdown and metrics for a single PDF."""
    page_count = _page_count(pdf_path)
    md_lines = [
        f"# OlmOCR (stub) — {pdf_path.name}",
        "",
        f"Pages: {page_count if page_count else 'unknown'}",
    ]
    if content_debug:
        md_lines.append("")
        md_lines.append("<!-- content_debug: stub output -->")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metrics = {"page_count": page_count}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model: Optional[str] = None,
    model_dir: Optional["Path"] = None,
    output_dir: Optional["Path"] = None,
    log_dir: Optional["Path"] = None,  # unused, mirrors other backend signatures
    max_pages: Optional[int] = None,  # reserved for future page-limit support
    allow_stub: bool = True,
    allow_cli: bool = False,
    allow_inproc: bool = True,
    allow_mlx_cli: bool = True,
    python_bin: Optional["Path"] = None,
    mlx_script: Optional["Path"] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,
    server: Optional[str] = None,
    api_key: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    target_longest_image_dim: Optional[int] = None,
    workers: Optional[int] = None,
    pages_per_group: Optional[int] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run OlmOCR-2 OCR for the provided files.

    Execution strategy (tried in order):

    1. In-process MLX if ``allow_inproc`` and ``mlx_vlm`` is importable (macOS).
    2. MLX CLI subprocess if ``allow_mlx_cli`` and the script exists (macOS).
    3. OlmOCR CLI subprocess if ``allow_cli`` (or ``GLOSSAPI_OLMOCR_ALLOW_CLI=1``).
    4. Stub output if ``allow_stub`` (and ``GLOSSAPI_OLMOCR_ALLOW_STUB=1``).

    Returns a mapping of ``stem -> {"page_count": int}``.
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = (
        Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    ).resolve()
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ----- Env overrides -----
    env = os.environ
    env_allow_stub = env.get("GLOSSAPI_OLMOCR_ALLOW_STUB", "1") == "1"
    env_allow_cli = env.get("GLOSSAPI_OLMOCR_ALLOW_CLI", "0") == "1"
    env_python = env.get("GLOSSAPI_OLMOCR_PYTHON")
    env_model = env.get("GLOSSAPI_OLMOCR_MODEL", "").strip()
    env_model_dir = env.get("GLOSSAPI_OLMOCR_MODEL_DIR", "").strip()
    env_mlx_model_dir = env.get("GLOSSAPI_OLMOCR_MLX_MODEL_DIR", "").strip()
    env_server = env.get("GLOSSAPI_OLMOCR_SERVER", "").strip()
    env_api_key = env.get("GLOSSAPI_OLMOCR_API_KEY", "").strip()
    env_gpu_mem = env.get("GLOSSAPI_OLMOCR_GPU_MEMORY_UTILIZATION", "").strip()
    env_max_model_len = env.get("GLOSSAPI_OLMOCR_MAX_MODEL_LEN", "").strip()
    env_tp = env.get("GLOSSAPI_OLMOCR_TENSOR_PARALLEL_SIZE", "").strip()
    env_target_dim = env.get("GLOSSAPI_OLMOCR_TARGET_IMAGE_DIM", "").strip()
    env_workers = env.get("GLOSSAPI_OLMOCR_WORKERS", "").strip()
    env_pages_per_group = env.get("GLOSSAPI_OLMOCR_PAGES_PER_GROUP", "").strip()
    env_device = env.get("GLOSSAPI_OLMOCR_DEVICE", "").strip()

    use_cli = allow_cli or env_allow_cli
    use_stub = allow_stub and env_allow_stub

    # Resolve parameters (kwargs > env > defaults)
    if python_bin is None and env_python:
        python_bin = Path(env_python)
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)

    # Model: local dir takes precedence over HF identifier
    effective_model = model or env_model or DEFAULT_MODEL
    if model_dir is None and env_model_dir:
        model_dir = Path(env_model_dir)
    if model_dir and model_dir.is_dir():
        effective_model = str(model_dir)

    # MLX model dir (separate from CUDA model)
    mlx_model_dir: Optional[Path] = None
    if env_mlx_model_dir:
        mlx_model_dir = Path(env_mlx_model_dir)
    else:
        # Fallback: check GLOSSAPI_WEIGHTS_ROOT/olmocr-mlx/
        from glossapi.ocr.utils.weights import resolve_weights_dir as _resolve_wdir
        mlx_model_dir = _resolve_wdir("olmocr-mlx")

    device_to_use = device or env_device or (
        "mps" if platform.system() == "Darwin" else "cuda"
    )

    effective_server = server or env_server or None
    effective_api_key = api_key or env_api_key or None

    effective_gpu_mem = gpu_memory_utilization
    if effective_gpu_mem is None and env_gpu_mem:
        try:
            effective_gpu_mem = float(env_gpu_mem)
        except ValueError:
            pass

    effective_max_model_len = max_model_len
    if effective_max_model_len is None and env_max_model_len:
        try:
            effective_max_model_len = int(env_max_model_len)
        except ValueError:
            pass

    effective_tp = tensor_parallel_size
    if effective_tp is None and env_tp:
        try:
            effective_tp = int(env_tp)
        except ValueError:
            pass

    effective_target_dim = target_longest_image_dim
    if effective_target_dim is None and env_target_dim:
        try:
            effective_target_dim = int(env_target_dim)
        except ValueError:
            pass

    effective_workers = workers
    if effective_workers is None and env_workers:
        try:
            effective_workers = int(env_workers)
        except ValueError:
            pass

    effective_pages_per_group = pages_per_group
    if effective_pages_per_group is None and env_pages_per_group:
        try:
            effective_pages_per_group = int(env_pages_per_group)
        except ValueError:
            pass

    # ----- Resolve input paths -----
    candidate_roots = _candidate_input_roots(input_root, out_root)
    resolved_paths, missing = _resolve_pdf_paths(file_list, candidate_roots)
    if missing:
        LOGGER.warning(
            "OlmOCR: %d input file(s) not found in candidate roots; OCR may be incomplete.",
            len(missing),
        )

    # ----- Strategy 1: In-process MLX (macOS Apple Silicon) -----
    if (
        allow_inproc
        and platform.system() == "Darwin"
        and _can_import_mlx()
    ):
        try:
            LOGGER.info("OlmOCR: using in-process MLX execution")
            return _run_inproc(
                resolved_paths,
                file_list,
                out_root,
                md_dir,
                metrics_dir,
                model_dir=mlx_model_dir or model_dir,
                max_pages=max_pages,
                content_debug=content_debug,
            )
        except Exception as exc:
            LOGGER.warning(
                "OlmOCR in-process MLX execution failed (%s); trying next strategy",
                exc,
            )
            if not allow_mlx_cli and not use_cli and not use_stub:
                raise

    # ----- Strategy 2: MLX CLI subprocess (macOS, separate venv) -----
    mlx_script_path = (
        _resolve_mlx_cli_script() if mlx_script is None else Path(mlx_script)
    )
    if allow_mlx_cli and platform.system() == "Darwin" and mlx_script_path.exists():
        cli_input_root = _pick_cli_input_root(
            file_list, resolved_paths, candidate_roots
        )
        try:
            _run_mlx_cli(
                cli_input_root,
                out_root,
                python_bin=python_bin,
                script=mlx_script_path,
                model_dir=mlx_model_dir or model_dir,
                max_pages=max_pages,
                content_debug=content_debug,
                device=device_to_use,
            )
            # Collect results from MLX CLI output
            results: Dict[str, Any] = {}
            for name, pdf_path in zip(file_list, resolved_paths):
                stem = Path(name).stem
                md_path = md_dir / f"{stem}.md"
                metrics_path = metrics_dir / f"{stem}.metrics.json"
                if not md_path.exists() or not md_path.read_text(encoding="utf-8").strip():
                    placeholder = [
                        f"# OlmOCR-2 — {pdf_path.name}",
                        "",
                        "[[Blank page]]",
                    ]
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    md_path.write_text("\n".join(placeholder) + "\n", encoding="utf-8")
                page_count = _page_count(pdf_path)
                if not metrics_path.exists():
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_path.write_text(
                        json.dumps({"page_count": page_count}, indent=2),
                        encoding="utf-8",
                    )
                results[stem] = {"page_count": page_count}
            return results
        except Exception as exc:
            if not use_cli and not use_stub:
                raise
            LOGGER.warning(
                "OlmOCR MLX CLI failed (%s); trying next strategy", exc,
            )

    # ----- Strategy 3: OlmOCR CLI subprocess (CUDA / vLLM) -----
    if use_cli:
        # OlmOCR uses its own workspace directory with a specific layout
        olmocr_workspace = out_root / "olmocr_workspace"
        olmocr_workspace.mkdir(parents=True, exist_ok=True)
        workspace_md_dir = olmocr_workspace / "markdown"

        existing_pdfs = [p for p in resolved_paths if p.exists()]
        if not existing_pdfs:
            LOGGER.warning("OlmOCR: no existing PDF files found; skipping CLI")
        else:
            try:
                _run_cli(
                    python_bin=python_exe,
                    workspace=olmocr_workspace,
                    pdf_paths=existing_pdfs,
                    model=effective_model,
                    server=effective_server,
                    api_key=effective_api_key,
                    gpu_memory_utilization=effective_gpu_mem,
                    max_model_len=effective_max_model_len,
                    tensor_parallel_size=effective_tp,
                    target_longest_image_dim=effective_target_dim,
                    workers=effective_workers,
                    pages_per_group=effective_pages_per_group,
                )
                return _collect_cli_results(
                    file_list,
                    resolved_paths,
                    workspace_md_dir,
                    md_dir,
                    metrics_dir,
                )
            except Exception as exc:
                if not use_stub:
                    raise
                LOGGER.warning("OlmOCR CLI failed (%s); falling back to stub output", exc)

    elif not use_stub:
        raise RuntimeError(
            "OlmOCR: no execution strategy available "
            "(in-process, MLX CLI, OlmOCR CLI, and stub are all disabled or failed)"
        )

    # ----- Strategy 4: Stub -----
    results: Dict[str, Any] = {}
    for name, pdf_path in zip(file_list, resolved_paths):
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        results[stem] = _write_stub(pdf_path, md_path, metrics_path, content_debug)
    return results
