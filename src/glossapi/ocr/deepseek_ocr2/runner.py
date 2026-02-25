"""DeepSeek OCR v2 runner — in-process MLX execution with CLI subprocess fallback.

The runner tries the following strategies in order:

1. **In-process** (``enable_inproc=True``, default on macOS when ``mlx_vlm`` is importable):
   Load the model once via :pymod:`glossapi.ocr.deepseek_ocr2.mlx_cli` and process
   every PDF without spawning a subprocess.  This is the fast path — the model stays
   loaded in memory across files.

2. **CLI subprocess** (``enable_ocr=True``):
   Shell out to ``python -m glossapi.ocr.deepseek_ocr2.mlx_cli`` (or a user-specified
   script via ``GLOSSAPI_DEEPSEEK2_MLX_SCRIPT``).  Useful when the main venv lacks
   ``mlx-vlm`` but a separate venv does.

3. **Stub** (``enable_stub=True``):
   Emit placeholder markdown + metrics.  Useful for dry-runs and testing.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from glossapi.ocr.utils.page import _page_count

LOGGER = logging.getLogger(__name__)

# Resolve the embedded CLI script shipped with the package.
_PACKAGE_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


def _candidate_input_roots(input_root: Path, output_root: Path) -> List[Path]:
    return [input_root, output_root / "downloads", input_root / "downloads"]


def _resolve_pdf_paths(
    file_list: Sequence[str],
    candidate_roots: Sequence[Path],
) -> Tuple[List[Path], List[str]]:
    resolved: List[Path] = []
    missing: List[str] = []
    for name in file_list:
        raw = Path(name)
        if raw.is_absolute():
            if not raw.exists():
                missing.append(name)
            resolved.append(raw)
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


# ---------------------------------------------------------------------------
# Strategy 1: In-process MLX execution
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
    LOGGER.info("Loading DeepSeek OCR v2 model from %s", model_path)
    model, processor = mlx_cli.load_model_and_processor(model_path)

    results: Dict[str, Any] = {}
    total_files = len(file_list)
    for file_idx, (name, pdf_path) in enumerate(zip(file_list, resolved_paths), 1):
        stem = Path(name).stem
        if not pdf_path.exists():
            LOGGER.warning("DeepSeek OCR v2: PDF not found: %s", pdf_path)
            results[stem] = {"page_count": 0}
            continue
        LOGGER.info("DeepSeek OCR v2: processing file %d/%d — %s", file_idx, total_files, pdf_path.name)
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
            LOGGER.error("DeepSeek OCR v2 in-process failed for %s: %s", pdf_path.name, exc)
            raise
    return results


# ---------------------------------------------------------------------------
# Strategy 2: CLI subprocess
# ---------------------------------------------------------------------------

def _resolve_cli_script() -> Path:
    """Return the path to the CLI script to invoke as a subprocess.

    Priority: ``GLOSSAPI_DEEPSEEK2_MLX_SCRIPT`` env var > package-embedded script.
    """
    env_script = os.environ.get("GLOSSAPI_DEEPSEEK2_MLX_SCRIPT", "").strip()
    if env_script:
        p = Path(env_script)
        if p.exists():
            return p
        LOGGER.warning("GLOSSAPI_DEEPSEEK2_MLX_SCRIPT=%s does not exist; using package script", env_script)
    return _PACKAGE_CLI_SCRIPT


def _run_cli(
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
    LOGGER.info("Running DeepSeek OCR v2 CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


# ---------------------------------------------------------------------------
# Strategy 3: Stub
# ---------------------------------------------------------------------------

def _run_one_pdf_stub(pdf_path: Path, md_out: Path, metrics_out: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stub processor for a single PDF — produces placeholder output."""
    page_count = _page_count(pdf_path)
    max_pages = cfg.get("max_pages")
    if max_pages is not None and page_count:
        page_count = min(page_count, max_pages)

    md_lines = [
        f"# DeepSeek OCR v2 (stub) - {pdf_path.name}",
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # unused, mirrors rapidocr signature
    max_pages: Optional[int] = None,
    enable_stub: bool = True,
    enable_ocr: bool = True,
    enable_inproc: bool = True,
    python_bin: Optional[Path] = None,
    mlx_script: Optional[Path] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR v2 for the provided files.

    Execution strategy (tried in order):

    1. In-process if ``enable_inproc`` and ``mlx_vlm`` is importable.
    2. CLI subprocess if ``enable_ocr`` and the script exists.
    3. Stub output if ``enable_stub``.

    Returns a mapping of ``stem -> {"page_count": int}``.
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ----- Resolve env overrides -----
    env = os.environ
    env_enable_stub = env.get("GLOSSAPI_DEEPSEEK2_ENABLE_STUB", "1") == "1"
    env_cli_override = env.get("GLOSSAPI_DEEPSEEK2_ENABLE_OCR")
    env_enable_ocr = env_cli_override == "1"
    env_python = env.get("GLOSSAPI_DEEPSEEK2_PYTHON")
    env_model_dir = env.get("GLOSSAPI_DEEPSEEK2_MODEL_DIR")
    env_device = env.get("GLOSSAPI_DEEPSEEK2_DEVICE")

    use_stub = enable_stub and env_enable_stub

    if python_bin is None and env_python:
        python_bin = Path(env_python)
    if model_dir is None and env_model_dir:
        model_dir = Path(env_model_dir)
    # Fallback: check GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-mlx/
    if model_dir is None:
        from glossapi.ocr.utils.weights import resolve_weights_dir
        model_dir = resolve_weights_dir("deepseek-ocr-mlx")

    device_to_use = device or env_device or ("mps" if platform.system() == "Darwin" else "cpu")

    # ----- Resolve input paths -----
    candidate_roots = _candidate_input_roots(input_root, out_root)
    resolved_paths, missing = _resolve_pdf_paths(file_list, candidate_roots)
    if missing:
        LOGGER.warning(
            "DeepSeek OCR v2: %d input file(s) not found in candidate roots; OCR may be incomplete.",
            len(missing),
        )

    # ----- Strategy 1: In-process -----
    if enable_inproc and platform.system() == "Darwin" and _can_import_mlx():
        try:
            LOGGER.info("DeepSeek OCR v2: using in-process MLX execution")
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
            LOGGER.warning("DeepSeek OCR v2 in-process execution failed (%s); trying next strategy", exc)
            if not enable_ocr and not use_stub:
                raise

    # ----- Strategy 2: CLI subprocess -----
    use_cli = enable_ocr or env_enable_ocr
    script_path = _resolve_cli_script() if mlx_script is None else Path(mlx_script)

    if use_cli and script_path.exists():
        if platform.system() != "Darwin":
            msg = "DeepSeek OCR v2 CLI requested on non-macOS"
            if not use_stub:
                raise RuntimeError(f"{msg}; stub fallback is disabled")
            LOGGER.warning("%s; falling back to stub output", msg)
        else:
            cli_input_root = _pick_cli_input_root(file_list, resolved_paths, candidate_roots)
            try:
                _run_cli(
                    cli_input_root,
                    out_root,
                    python_bin=python_bin,
                    script=script_path,
                    model_dir=model_dir,
                    max_pages=max_pages,
                    content_debug=content_debug,
                    device=device_to_use,
                )
                # Collect results after CLI run
                results: Dict[str, Any] = {}
                for name, pdf_path in zip(file_list, resolved_paths):
                    stem = Path(name).stem
                    md_path = md_dir / f"{stem}.md"
                    metrics_path = metrics_dir / f"{stem}.metrics.json"
                    if not md_path.exists() or not md_path.read_text(encoding="utf-8").strip():
                        placeholder = [
                            f"# DeepSeek OCR v2 - {pdf_path.name}",
                            "",
                            "[[Blank page]]",
                        ]
                        md_path.parent.mkdir(parents=True, exist_ok=True)
                        md_path.write_text("\n".join(placeholder) + "\n", encoding="utf-8")
                    page_count = _page_count(pdf_path)
                    if not metrics_path.exists():
                        metrics_path.parent.mkdir(parents=True, exist_ok=True)
                        metrics_path.write_text(json.dumps({"page_count": page_count}, indent=2), encoding="utf-8")
                    results[stem] = {"page_count": page_count}
                return results
            except Exception as exc:
                if not use_stub:
                    raise
                LOGGER.warning("DeepSeek OCR v2 CLI failed (%s); falling back to stub output", exc)

    # ----- Strategy 3: Stub -----
    if not use_stub:
        raise RuntimeError(
            "DeepSeek OCR v2: no execution strategy available (in-process, CLI, and stub are all disabled or failed)"
        )

    LOGGER.info("DeepSeek OCR v2: using stub output for %d file(s)", len(file_list))
    cfg = {"max_pages": max_pages, "content_debug": content_debug}
    results = {}
    for name, pdf_path in zip(file_list, resolved_paths):
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        results[stem] = _run_one_pdf_stub(pdf_path, md_path, metrics_path, cfg)
    return results
