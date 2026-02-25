"""DeepSeek OCR runner with stub and optional CLI dispatch."""

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

from glossapi.ocr.utils.page import _page_count

LOGGER = logging.getLogger(__name__)

# Embedded MLX CLI script shipped with the package.
_PACKAGE_MLX_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


def _run_cli_vllm(
    input_dir: Path,
    output_dir: Path,
    *,
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    gpu_memory_utilization: Optional[float] = None,
    disable_fp8_kv: bool = False,
) -> None:
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)
    cmd: List[str] = [
        str(python_exe),
        str(script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if disable_fp8_kv:
        cmd.append("--no-fp8-kv")

    env = os.environ.copy()
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        # FlashInfer JIT (via vLLM) needs a C++ toolchain; add a known cc1plus location if missing.
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH','')}"
            break
    ld_path = env.get("GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH")
    if ld_path:
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH','')}"

    LOGGER.info("Running DeepSeek OCR vLLM CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


# ---------------------------------------------------------------------------
# MLX / MPS helpers (Apple Silicon)
# ---------------------------------------------------------------------------

def _can_import_mlx() -> bool:
    """Return True if mlx-vlm is importable in the current process."""
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_mlx_cli_script() -> Path:
    """Return the path to the MLX CLI script.

    Priority: ``GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT`` env var > package-embedded script.
    """
    env_script = os.environ.get("GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT", "").strip()
    if env_script:
        p = Path(env_script)
        if p.exists():
            return p
        LOGGER.warning(
            "GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT=%s does not exist; using package script", env_script
        )
    return _PACKAGE_MLX_CLI_SCRIPT


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
    """Process PDFs in-process using the embedded mlx_cli module (MPS fast path)."""
    from . import mlx_cli

    model_path = mlx_cli.resolve_model_dir(str(model_dir) if model_dir else None)
    LOGGER.info("Loading DeepSeek OCR MLX model from %s", model_path)
    model, processor = mlx_cli.load_model_and_processor(model_path)

    results: Dict[str, Any] = {}
    total_files = len(file_list)
    for file_idx, (name, pdf_path) in enumerate(zip(file_list, resolved_paths), 1):
        stem = Path(name).stem
        if not pdf_path.exists():
            LOGGER.warning("DeepSeek OCR MLX: PDF not found: %s", pdf_path)
            results[stem] = {"page_count": 0}
            continue
        LOGGER.info(
            "DeepSeek OCR MLX: processing file %d/%d — %s", file_idx, total_files, pdf_path.name
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
            LOGGER.error("DeepSeek OCR MLX in-process failed for %s: %s", pdf_path.name, exc)
            raise
    return results


def _run_cli_mlx(
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
    """Invoke the MLX CLI as a subprocess (MPS subprocess path)."""
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
    LOGGER.info("Running DeepSeek OCR MLX CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


def _run_one_pdf(pdf_path: Path, md_out: Path, metrics_out: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stub processor for a single PDF."""
    page_count = _page_count(pdf_path)
    max_pages = cfg.get("max_pages")
    if max_pages is not None and page_count:
        page_count = min(page_count, max_pages)

    md_lines = [
        f"# DeepSeek OCR (stub) — {pdf_path.name}",
        "",
        f"Pages: {page_count if page_count else 'unknown'}",
    ]
    if cfg.get("content_debug"):
        md_lines.append("")
        md_lines.append("<!-- content_debug: stub output -->")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metrics = {"page_count": page_count}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _collect_cli_results(
    file_list: List[str],
    input_root: Path,
    md_dir: Path,
    metrics_dir: Path,
    *,
    backend_label: str,
) -> Dict[str, Any]:
    """Gather per-stem results after any CLI run that wrote markdown + metrics."""
    results: Dict[str, Any] = {}
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        if not md_path.exists() or not md_path.read_text(encoding="utf-8").strip():
            placeholder = [f"# {backend_label} — {pdf_path.name}", "", "[[Blank page]]"]
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("\n".join(placeholder) + "\n", encoding="utf-8")
        page_count = _page_count(pdf_path)
        if not metrics_path.exists():
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps({"page_count": page_count}, indent=2), encoding="utf-8"
            )
        results[stem] = {"page_count": page_count}
    return results


# ---------------------------------------------------------------------------
# Path resolution helpers (multi-root, same pattern as deepseek_ocr2 / glm_ocr)
# ---------------------------------------------------------------------------


def _candidate_input_roots(input_root: Path, output_root: Path) -> List[Path]:
    """Return directories to search for input PDFs, in priority order."""
    return [input_root, output_root / "downloads", input_root / "downloads"]


def _resolve_pdf_paths(
    file_list: Sequence[str],
    candidate_roots: Sequence[Path],
) -> Tuple[List[Path], List[str]]:
    """Resolve each filename to an absolute path by searching candidate roots.

    Returns ``(resolved_paths, missing)`` where *missing* contains names that
    could not be found in any candidate root.
    """
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


def _pick_cli_input_root(
    file_list: Sequence[str],
    resolved_paths: Sequence[Path],
    candidate_roots: Sequence[Path],
) -> Path:
    """Pick the best input root directory for a CLI subprocess invocation.

    If all resolved paths share a single parent directory that root is used.
    Otherwise the candidate root that resolves the most filenames wins.
    """
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


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # unused placeholder to mirror rapidocr
    max_pages: Optional[int] = None,
    enable_stub: bool = True,
    enable_ocr: bool = False,
    enable_inproc: bool = True,
    enable_mlx_ocr: bool = True,
    python_bin: Optional[Path] = None,
    vllm_script: Optional[Path] = None,
    mlx_script: Optional[Path] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    disable_fp8_kv: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR for the provided files.

    Execution strategy (in order):

    **macOS / MPS path:**

    1. In-process MLX (``enable_inproc=True``, macOS, ``mlx_vlm`` importable).
    2. MLX CLI subprocess (``enable_mlx_ocr=True``, macOS, script present).

    **CUDA path (Linux / Windows):**

    3. vLLM CLI subprocess (``enable_ocr=True``, script present).

    **Fallback (both paths):**

    4. Stub output (``enable_stub=True``).

    The active device is resolved from ``GLOSSAPI_DEEPSEEK_OCR_DEVICE`` env var,
    then the *device* kwarg, then auto-detected (``mps`` on macOS, ``cuda`` otherwise).

    Returns a mapping of ``stem -> {"page_count": int}``.
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = (
        Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    )
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ----- Env overrides -----
    env = os.environ
    env_enable_stub = env.get("GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB", "1") == "1"
    # GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR controls the CUDA/vLLM CLI path.
    env_enable_ocr = env.get("GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR", "0") == "1"
    # GLOSSAPI_DEEPSEEK_OCR_ENABLE_MLX_OCR controls the MPS/MLX CLI subprocess path
    # independently of the CUDA CLI flag.
    env_enable_mlx_ocr: Optional[str] = env.get("GLOSSAPI_DEEPSEEK_OCR_ENABLE_MLX_OCR")
    env_device = env.get("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "").strip().lower()
    # Accept both names: GLOSSAPI_DEEPSEEK_OCR_PYTHON (consistent with other backends)
    # and the legacy GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON.
    env_python = (
        env.get("GLOSSAPI_DEEPSEEK_OCR_PYTHON", "")
        or env.get("GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON", "")
    )
    env_mlx_model_dir = env.get("GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR", "")
    env_gpu_mem = env.get("GLOSSAPI_DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION", "")

    use_stub = enable_stub and env_enable_stub
    if python_bin is None and env_python:
        python_bin = Path(env_python)
    if model_dir is None and env_mlx_model_dir:
        model_dir = Path(env_mlx_model_dir)
    # Fallback: locate weights under GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-1-mlx/
    if model_dir is None:
        try:
            from glossapi.ocr.utils.weights import resolve_weights_dir as _rwd
            _resolved = _rwd("deepseek-ocr-1-mlx")
            if _resolved is not None:
                model_dir = _resolved
        except Exception:
            pass

    gpu_mem_fraction = gpu_memory_utilization
    if env_gpu_mem:
        try:
            gpu_mem_fraction = float(env_gpu_mem)
        except Exception:
            pass
    disable_fp8_kv = disable_fp8_kv or env.get("GLOSSAPI_DEEPSEEK_OCR_NO_FP8_KV") == "1"

    # ----- Device detection -----
    # Priority: env var > kwarg > auto-detect
    is_macos = platform.system() == "Darwin"
    if env_device:
        active_device = env_device
    elif device:
        active_device = device.lower()
    else:
        active_device = "mps" if is_macos else "cuda"

    # Explicitly requested "cpu" on macOS must not be coerced into MPS.
    use_mps = active_device == "mps" or (is_macos and active_device not in ("cuda", "cpu"))

    # ----- Resolve input paths (multi-root search) -----
    candidate_roots = _candidate_input_roots(input_root, out_root)
    resolved_paths, missing_paths = _resolve_pdf_paths(file_list, candidate_roots)
    if missing_paths:
        LOGGER.warning(
            "DeepSeek OCR: %d input file(s) not found in candidate roots; OCR may be incomplete.",
            len(missing_paths),
        )

    # =========================================================================
    # MPS / MLX path — Apple Silicon
    # =========================================================================
    if use_mps:
        # ----- Strategy 1: In-process MLX -----
        if enable_inproc and is_macos and _can_import_mlx():
            try:
                LOGGER.info("DeepSeek OCR: using in-process MLX execution (MPS)")
                return _run_inproc(
                    resolved_paths,
                    file_list,
                    out_root,
                    md_dir,
                    metrics_dir,
                    model_dir=model_dir,
                    max_pages=max_pages,
                    content_debug=content_debug,
                )
            except Exception as exc:
                LOGGER.warning(
                    "DeepSeek OCR in-process MLX failed (%s); trying next strategy", exc
                )
                if not enable_mlx_ocr and not use_stub:
                    raise

        # ----- Strategy 2: MLX CLI subprocess -----
        mlx_script_path = (
            _resolve_mlx_cli_script() if mlx_script is None else Path(mlx_script)
        )
        # env var wins over kwarg; allows enabling the MLX CLI from the environment
        # without touching code (mirrors GLOSSAPI_DEEPSEEK2_ENABLE_OCR for V2).
        use_mlx_cli = enable_mlx_ocr
        if env_enable_mlx_ocr is not None:
            use_mlx_cli = env_enable_mlx_ocr == "1"
        if use_mlx_cli and mlx_script_path.exists():
            if not is_macos:
                msg = "DeepSeek OCR MLX CLI requested on non-macOS"
                if not use_stub:
                    raise RuntimeError(f"{msg}; stub fallback is disabled")
                LOGGER.warning("%s; falling back to stub output", msg)
            else:
                try:
                    cli_input_root = _pick_cli_input_root(
                        file_list, resolved_paths, candidate_roots
                    )
                    _run_cli_mlx(
                        cli_input_root,
                        out_root,
                        python_bin=python_bin,
                        script=mlx_script_path,
                        model_dir=model_dir,
                        max_pages=max_pages,
                        content_debug=content_debug,
                        device=active_device,
                    )
                    return _collect_cli_results(
                        file_list, cli_input_root, md_dir, metrics_dir,
                        backend_label="DeepSeek OCR",
                    )
                except Exception as exc:
                    if not use_stub:
                        raise
                    LOGGER.warning(
                        "DeepSeek OCR MLX CLI failed (%s); falling back to stub output", exc
                    )

    # =========================================================================
    # CUDA / vLLM path
    # =========================================================================
    else:
        vllm_script_path = (
            Path(vllm_script)
            if vllm_script
            else Path.cwd() / "deepseek-ocr" / "run_pdf_ocr_vllm.py"
        )
        use_cli = enable_ocr or env_enable_ocr
        if use_cli and vllm_script_path.exists():
            try:
                _run_cli_vllm(
                    input_root,
                    out_root,
                    python_bin=python_bin,
                    script=vllm_script_path,
                    max_pages=max_pages,
                    content_debug=content_debug,
                    gpu_memory_utilization=gpu_mem_fraction,
                    disable_fp8_kv=disable_fp8_kv,
                )
                return _collect_cli_results(
                    file_list, input_root, md_dir, metrics_dir,
                    backend_label="DeepSeek OCR",
                )
            except Exception as exc:
                if not use_stub:
                    from glossapi.ocr.utils.cuda import is_cuda_setup_error, raise_cuda_diagnosis
                    if is_cuda_setup_error(exc):
                        python_exe = Path(python_bin) if python_bin else Path(sys.executable)
                        raise_cuda_diagnosis(
                            "deepseek-ocr",
                            [("DeepSeek vLLM CLI", exc)],
                            python_exe,
                        )
                    raise
                LOGGER.warning(
                    "DeepSeek OCR vLLM CLI failed (%s); falling back to stub output", exc
                )

    # =========================================================================
    # Stub fallback
    # =========================================================================
    if not use_stub:
        raise RuntimeError(
            "DeepSeek OCR: no execution strategy available "
            "(in-process, CLI, and stub are all disabled or failed)"
        )

    LOGGER.info("DeepSeek OCR: using stub output for %d file(s)", len(file_list))
    cfg = {"max_pages": max_pages, "content_debug": content_debug}
    results: Dict[str, Any] = {}
    for name, pdf_path in zip(file_list, resolved_paths):
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        results[stem] = _run_one_pdf(pdf_path, md_path, metrics_path, cfg)
    return results
