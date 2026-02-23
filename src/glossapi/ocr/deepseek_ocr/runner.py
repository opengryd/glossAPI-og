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
from typing import Any, Dict, Iterable, List, Optional

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)

# Embedded MLX CLI script shipped with the package.
_PACKAGE_MLX_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
    except Exception:
        return 0


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


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # unused placeholder to mirror rapidocr
    max_pages: Optional[int] = None,
    allow_stub: bool = True,
    allow_cli: bool = False,
    allow_inproc: bool = True,
    allow_mlx_cli: bool = True,
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

    1. In-process MLX (``allow_inproc=True``, macOS, ``mlx_vlm`` importable).
    2. MLX CLI subprocess (``allow_mlx_cli=True``, macOS, script present).

    **CUDA path (Linux / Windows):**

    3. vLLM CLI subprocess (``allow_cli=True``, script present).

    **Fallback (both paths):**

    4. Stub output (``allow_stub=True``).

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
    env_allow_stub = env.get("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1") == "1"
    env_allow_cli = env.get("GLOSSAPI_DEEPSEEK_OCR_ALLOW_CLI", "0") == "1"
    env_device = env.get("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "").strip().lower()
    env_python = env.get("GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON", "")
    env_mlx_model_dir = env.get("GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR", "")
    env_gpu_mem = env.get("GLOSSAPI_DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION", "")

    use_stub = allow_stub and env_allow_stub
    if python_bin is None and env_python:
        python_bin = Path(env_python)
    if model_dir is None and env_mlx_model_dir:
        model_dir = Path(env_mlx_model_dir)

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

    use_mps = active_device == "mps" or (active_device not in ("cuda",) and is_macos)

    # ----- Resolve input paths -----
    resolved_paths: List[Path] = []
    for name in file_list:
        raw = Path(name)
        if raw.is_absolute():
            resolved_paths.append(raw)
        else:
            candidate = (input_root / name).resolve()
            if not candidate.exists():
                dl_candidate = (out_root / "downloads" / name).resolve()
                candidate = dl_candidate if dl_candidate.exists() else candidate
            resolved_paths.append(candidate)

    # =========================================================================
    # MPS / MLX path — Apple Silicon
    # =========================================================================
    if use_mps:
        # ----- Strategy 1: In-process MLX -----
        if allow_inproc and is_macos and _can_import_mlx():
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
                if not allow_mlx_cli and not use_stub:
                    raise

        # ----- Strategy 2: MLX CLI subprocess -----
        mlx_script_path = (
            _resolve_mlx_cli_script() if mlx_script is None else Path(mlx_script)
        )
        use_mlx_cli = allow_mlx_cli
        if use_mlx_cli and mlx_script_path.exists():
            if not is_macos:
                msg = "DeepSeek OCR MLX CLI requested on non-macOS"
                if not use_stub:
                    raise RuntimeError(f"{msg}; stub fallback is disabled")
                LOGGER.warning("%s; falling back to stub output", msg)
            else:
                try:
                    _run_cli_mlx(
                        input_root,
                        out_root,
                        python_bin=python_bin,
                        script=mlx_script_path,
                        model_dir=model_dir,
                        max_pages=max_pages,
                        content_debug=content_debug,
                        device=active_device,
                    )
                    return _collect_cli_results(
                        file_list, input_root, md_dir, metrics_dir,
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
        use_cli = allow_cli or env_allow_cli
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
