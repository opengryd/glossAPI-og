#!/usr/bin/env python3
"""GLM-OCR MLX CLI — standalone script for in-process and subprocess invocation.

This module can be run directly as a script
(``python -m glossapi.ocr.glm_ocr.mlx_cli``) or imported for in-process model
loading and PDF processing on Apple Silicon via MLX.

GLM-OCR (https://huggingface.co/mlx-community/GLM-OCR-4bit) is a compact
0.5B-parameter VLM fine-tuned for document OCR.  The MLX-community quantised
variant runs efficiently on Apple Silicon.

Heavy dependencies (mlx-vlm, transformers, huggingface_hub) are imported
lazily so that importing the parent package remains lightweight.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred until actually needed
# ---------------------------------------------------------------------------

try:
    import pypdfium2 as pdfium
except Exception as exc:  # pragma: no cover
    pdfium = None
    _PDFIUM_ERROR = exc

try:
    from PIL import Image  # noqa: F401
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_ERROR = exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GLM-OCR supports exactly three fine-tuned prompt types.
# See https://huggingface.co/zai-org/GLM-OCR#prompt-limited
PROMPT_TEXT = "Text Recognition:"
PROMPT_FORMULA = "Formula Recognition:"
PROMPT_TABLE = "Table Recognition:"

# For backward compatibility and single-pass callers.
DEFAULT_PROMPT = PROMPT_TEXT

# All three prompts for maximum-quality multi-pass OCR.
MULTI_PASS_PROMPTS = (PROMPT_TEXT, PROMPT_FORMULA, PROMPT_TABLE)

DEFAULT_DPI = 150
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MODEL_ID = "mlx-community/GLM-OCR-4bit"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


# Minimum mlx-vlm version required for glm_ocr model type support.
_MIN_MLX_VLM_VERSION = "0.3.12"
_MIN_TRANSFORMERS_VERSION = "5.1.0"


def _check_mlx_vlm_version() -> None:
    """Verify that the installed mlx-vlm version supports glm_ocr."""
    try:
        import mlx_vlm
        version = getattr(mlx_vlm, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(version) < Version(_MIN_MLX_VLM_VERSION):
            raise ImportError(
                f"mlx-vlm>={_MIN_MLX_VLM_VERSION} is required for GLM-OCR "
                f"(glm_ocr model type), but {version} is installed. "
                f"Run: pip install --upgrade mlx-vlm"
            )
    except ImportError as exc:
        if "mlx-vlm" in str(exc) or "packaging" in str(exc):
            raise
        # mlx_vlm not installed at all — load() will fail with a clear message


def _check_transformers_version() -> None:
    """Verify that transformers is new enough for GlmOcrProcessor."""
    try:
        import transformers
        version = getattr(transformers, "__version__", "0.0.0")
        from packaging.version import Version
        if Version(version) < Version(_MIN_TRANSFORMERS_VERSION):
            raise ImportError(
                f"transformers>={_MIN_TRANSFORMERS_VERSION} is required for "
                f"GLM-OCR (GlmOcrProcessor), but {version} is installed. "
                f"Run: pip install --no-deps 'transformers>={_MIN_TRANSFORMERS_VERSION}'"
            )
    except ImportError as exc:
        if "transformers" in str(exc) or "packaging" in str(exc):
            raise


def load_model_and_processor(model_path: Path) -> tuple:
    """Load the MLX model and processor for GLM-OCR.

    Sets the Metal wired memory limit before weight allocation to prevent
    unified-memory page-outs, then compiles Metal shaders with a dummy
    inference so the first real page does not pay the JIT cost.

    GLM-OCR requires ``mlx-vlm >= 0.3.12`` for native ``glm_ocr`` model type
    support.

    Returns ``(model, processor)`` ready for :func:`generate_page`.
    """
    _set_metal_wired_limit()
    model, processor = _load_impl(model_path)
    _warmup_metal_shaders(model, processor)
    return model, processor


def _load_impl(model_path: Path) -> tuple:
    """Internal: raw model + processor load with tokenizer-class fallback."""
    _check_mlx_vlm_version()
    _check_transformers_version()
    from mlx_vlm import load

    try:
        return load(str(model_path), trust_remote_code=True)
    except ValueError as exc:
        if "Unrecognized" not in str(exc):
            raise

        # Fallback: load model and tokenizer separately
        from mlx_vlm.utils import load_model as _load_model
        from transformers import AutoProcessor, AutoTokenizer

        model = _load_model(model_path, trust_remote_code=True)

        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        except Exception:
            # Last resort — construct processor from tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
            processor = tokenizer

        return model, processor


def _try_hf_cache(model_id: str) -> Optional[Path]:
    """Check whether *model_id* already exists in the HuggingFace Hub cache.

    Returns the snapshot directory if found, otherwise ``None``.
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                # Pick the most recent revision
                revisions = sorted(
                    repo.revisions,
                    key=lambda r: r.last_modified,
                    reverse=True,
                )
                if revisions:
                    snap = revisions[0].snapshot_path
                    # Sanity: the snapshot must contain at least one safetensors
                    if any(snap.glob("*.safetensors")):
                        return snap
    except Exception:
        pass
    return None


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """Resolve the model directory: env var > explicit arg > weights root > HF cache > download.

    Priority:
    1. ``GLOSSAPI_GLMOCR_MODEL_DIR`` environment variable
    2. *model_dir* argument (local path or HuggingFace repo id)
    3. ``GLOSSAPI_WEIGHTS_ROOT/glm-ocr-mlx/`` if present on disk
    4. HuggingFace Hub cache (already-downloaded snapshot)
    5. ``GLOSSAPI_GLMOCR_MLX_MODEL`` env var as HuggingFace repo id
    6. Auto-download from ``mlx-community/GLM-OCR-4bit``

    When downloading, if ``GLOSSAPI_WEIGHTS_ROOT`` is set the model is saved
    directly into ``<root>/glm-ocr-mlx/`` so that subsequent runs resolve at
    step 3 without network access.
    """
    from huggingface_hub import snapshot_download
    from glossapi.ocr.utils.weights import resolve_weights_dir

    log = logging.getLogger(__name__)

    env_dir = (os.getenv("GLOSSAPI_GLMOCR_MODEL_DIR") or "").strip()
    if env_dir:
        env_path = Path(env_dir).expanduser()
        if env_path.exists():
            return env_path

    if model_dir:
        path = Path(model_dir).expanduser()
        if path.exists():
            return path
        model_id = str(model_dir)
    else:
        # Check GLOSSAPI_WEIGHTS_ROOT fallback
        resolved = resolve_weights_dir("glm-ocr-mlx")
        if resolved is not None:
            return resolved
        model_id = (
            os.getenv("GLOSSAPI_GLMOCR_MLX_MODEL", "").strip() or DEFAULT_MODEL_ID
        )

    # Check HuggingFace Hub cache before triggering a download
    cached = _try_hf_cache(model_id)
    if cached is not None:
        log.info("GLM-OCR MLX model found in HF cache: %s", cached)
        return cached

    # Determine download destination — prefer explicit env var, then
    # the project-default <repo>/model_weights/.
    from glossapi.ocr.utils.weights import default_weights_root

    weights_root_env = (os.getenv("GLOSSAPI_WEIGHTS_ROOT") or "").strip()
    weights_root = Path(weights_root_env) if weights_root_env else default_weights_root()

    download_kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "*.py",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ],
    }
    if weights_root is not None:
        # Download directly into the stable weights directory so
        # resolve_weights_dir() finds it on the next run.
        local_target = weights_root / "glm-ocr-mlx"
        local_target.mkdir(parents=True, exist_ok=True)
        download_kwargs["local_dir"] = str(local_target)
        print(f"Downloading model: {model_id} → {local_target}")
    else:
        print(f"Downloading model: {model_id}")

    local_dir = snapshot_download(**download_kwargs)
    return Path(local_dir)


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------


def _flush_metal() -> None:
    """Flush pending MLX computations and release cached Metal buffers.

    ``mx.eval()`` forces lazy evaluation of the computation graph, freeing
    intermediate activations.  ``mx.clear_cache()`` returns unretained
    Metal buffer slabs to the OS allocator.  Model weights (held as live Python
    objects) are never released by this call.
    """
    try:
        import mlx.core as mx  # noqa: PLC0415
        mx.eval()
        mx.clear_cache()
    except Exception:
        pass


def _set_metal_wired_limit(fraction: float = 0.8) -> None:
    """Hint to the MLX Metal allocator to wire a large fraction of device memory.

    Wiring model weights prevents the macOS memory compressor from reclaiming
    them during batch inference under unified-memory pressure.  Must be called
    *before* allocating model weights — only future allocations are affected.

    *fraction* is applied to ``recommendedMaxWorkingSetSize`` (the OS-reported
    safe upper bound for GPU allocations).  0.8 leaves headroom for the OS and
    other Metal processes.
    """
    try:
        import mlx.core as mx  # noqa: PLC0415
        info = mx.device_info()
        recommended = info.get("recommendedMaxWorkingSetSize", 0)
        if recommended > 0:
            mx.set_wired_limit(int(recommended * fraction))
    except Exception:
        pass


def render_page(doc: Any, page_index: int, dpi: int = DEFAULT_DPI) -> Any:
    """Render a single PDF page to a PIL Image via pypdfium2."""
    page = doc[page_index]
    scale = float(dpi) / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def generate_page(
    model: Any,
    processor: Any,
    image: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run inference on a single page image and return the raw text for one prompt."""
    from mlx_vlm import generate

    result = generate(
        model,
        processor,
        prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=0.0,
        repetition_penalty=1.05,
        skip_special_tokens=True,
    )
    # mlx_vlm.generate may return a string or an object with a .text attribute
    if isinstance(result, str):
        text = result.strip()
    else:
        text = (getattr(result, "text", None) or str(result)).strip()
    return text if text else ""


def _warmup_metal_shaders(model: Any, processor: Any) -> None:
    """Run a tiny dummy inference to pre-compile Metal shaders.

    MLX compiles Metal shaders JIT on first use.  For large models this makes
    the first real page 3–5× slower than subsequent pages.  A single forward
    pass on a 128×128 white image absorbs the compilation cost up front.
    """
    log = logging.getLogger(__name__)
    if Image is None:
        return
    try:
        dummy = Image.new("RGB", (128, 128), color=255)
        log.debug("Warming up Metal shaders (first-inference JIT)…")
        generate_page(model, processor, dummy, max_tokens=4)
        _flush_metal()
        log.debug("Metal shader warm-up complete.")
    except Exception as exc:
        log.debug("Metal shader warm-up skipped (%s).", exc)


def _is_blank(text: str) -> bool:
    """Return True if *text* is empty or a blank-page sentinel."""
    return not text or text == "[[Blank page]]"


def generate_page_multipass(
    model: Any,
    processor: Any,
    image: Any,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run all three GLM-OCR prompts on a page image and merge the results.

    Execution order:
    1. **Text Recognition** — primary content (text, headings, paragraphs).
    2. **Formula Recognition** — LaTeX equations.
    3. **Table Recognition** — markdown tables.

    The merge strategy appends formula and table content that is not already
    present in the text pass, avoiding duplication while ensuring nothing is
    lost.
    """
    text_out = generate_page(
        model, processor, image, prompt=PROMPT_TEXT, max_tokens=max_tokens,
    )
    _flush_metal()  # release text-pass activations before formula pass
    formula_out = generate_page(
        model, processor, image, prompt=PROMPT_FORMULA, max_tokens=max_tokens,
    )
    _flush_metal()  # release formula-pass activations before table pass
    table_out = generate_page(
        model, processor, image, prompt=PROMPT_TABLE, max_tokens=max_tokens,
    )

    return _merge_multipass(text_out, formula_out, table_out)


def _merge_multipass(text_out: str, formula_out: str, table_out: str) -> str:
    """Merge the outputs of the three GLM-OCR prompt passes.

    Strategy:
    - Start with the text pass as the base.
    - If the formula pass produced LaTeX content not already in the text pass,
      append it in a dedicated section.
    - If the table pass produced table content not already in the text pass,
      append it in a dedicated section.
    - If *all* passes are blank, return the blank-page sentinel.
    """
    parts: list[str] = []

    # --- Base text ---
    if not _is_blank(text_out):
        parts.append(text_out)

    # --- Formulas ---
    if not _is_blank(formula_out):
        # Only append if the formula content adds something new.
        # Simple heuristic: if the formula text is not a substring of the
        # already-collected text, include it.
        collected = "\n".join(parts)
        if formula_out not in collected:
            parts.append(formula_out)

    # --- Tables ---
    if not _is_blank(table_out):
        collected = "\n".join(parts)
        if table_out not in collected:
            parts.append(table_out)

    if not parts:
        return "[[Blank page]]"

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PDF-level processing
# ---------------------------------------------------------------------------


def _write_metrics(metrics_path: Path, page_count: int) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"page_count": int(page_count)}, indent=2), encoding="utf-8"
    )


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    model: Any,
    processor: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_pages: Optional[int] = None,
    dpi: int = DEFAULT_DPI,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    content_debug: bool = False,
    multipass: bool = True,
) -> int:
    """Process a single PDF and write markdown + metrics files.

    When *multipass* is ``True`` (the default), every page is processed with
    all three GLM-OCR prompts (Text / Formula / Table) and the results are
    merged for maximum accuracy.  Set ``multipass=False`` to use only the
    single *prompt* (faster, lower quality).

    Returns the number of pages processed.
    """
    if pdfium is None:
        raise RuntimeError(f"pypdfium2 is required but unavailable: {_PDFIUM_ERROR}")
    if Image is None:
        raise RuntimeError(f"Pillow is required but unavailable: {_PIL_ERROR}")

    doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(doc)
    page_count = min(total_pages, max_pages) if max_pages else total_pages

    markdown_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_md = markdown_dir / f"{pdf_path.stem}.md"
    out_metrics = metrics_dir / f"{pdf_path.stem}.metrics.json"

    lines: list[str] = []
    logger = logging.getLogger(__name__)

    mode_label = "multi-pass" if multipass else "single-pass"
    passes_per_page = len(MULTI_PASS_PROMPTS) if multipass else 1

    # tqdm progress bar with logging fallback
    try:
        from tqdm import tqdm as _tqdm

        progress = _tqdm(
            total=page_count * passes_per_page,
            desc=f"OCR {pdf_path.name} ({mode_label})",
            unit="pass",
            dynamic_ncols=True,
        )
    except ImportError:
        progress = None

    pdf_start = time.time()
    with ThreadPoolExecutor(max_workers=1) as _render_pool:
        # Pre-render the first page to bootstrap the CPU/GPU pipeline.
        _next_render = _render_pool.submit(render_page, doc, 0, dpi)
        for page_index in range(page_count):
            if content_debug:
                lines.append(f"<!-- page:{page_index + 1} -->")
            image = _next_render.result()
            # Prefetch next page on CPU while Metal runs inference on current page.
            if page_index + 1 < page_count:
                _next_render = _render_pool.submit(render_page, doc, page_index + 1, dpi)

            if multipass:
                text = generate_page_multipass(
                    model, processor, image, max_tokens=max_tokens,
                )
                if progress is not None:
                    progress.update(passes_per_page)
            else:
                text = generate_page(
                    model, processor, image, prompt=prompt, max_tokens=max_tokens,
                )
                if not text:
                    text = "[[Blank page]]"
                if progress is not None:
                    progress.update(1)

            _flush_metal()  # flush Metal command buffer and release activation slabs
            lines.append(text)
            lines.append("")

            if progress is None:
                logger.info(
                    "GLM-OCR MLX (%s): %s page %d/%d done",
                    mode_label,
                    pdf_path.name,
                    page_index + 1,
                    page_count,
                )

    if progress is not None:
        progress.close()

    elapsed_total = time.time() - pdf_start
    logger.info(
        "GLM-OCR MLX (%s): %s complete — %d pages in %.1fs (%.2fs/page)",
        mode_label,
        pdf_path.name,
        page_count,
        elapsed_total,
        elapsed_total / max(page_count, 1),
    )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    _write_metrics(out_metrics, page_count)
    return page_count


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry-point for GLM-OCR MLX PDF runner."""
    parser = argparse.ArgumentParser(description="GLM-OCR (MLX) PDF runner")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--content-debug", action="store_true")
    parser.add_argument(
        "--device", default=None, help="Optional device hint (unused)."
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Override the OCR prompt (single-pass only)."
    )
    parser.add_argument(
        "--no-multipass",
        action="store_true",
        help="Disable multi-pass OCR (text+formula+table). Uses single prompt instead.",
    )

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    use_multipass = not args.no_multipass

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    model_path = resolve_model_dir(args.model_dir)
    model, processor = load_model_and_processor(model_path)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in input dir.")
        return 0

    mode_label = "multi-pass" if use_multipass else "single-pass"
    print(f"Processing {len(pdfs)} PDF(s) in {mode_label} mode")

    start = time.time()
    for pdf_path in pdfs:
        process_pdf(
            pdf_path,
            output_dir,
            model,
            processor,
            prompt=args.prompt,
            max_pages=args.max_pages,
            dpi=args.dpi,
            max_tokens=args.max_tokens,
            content_debug=bool(args.content_debug),
            multipass=use_multipass,
        )
    elapsed = time.time() - start
    print(f"Processed {len(pdfs)} PDF(s) in {elapsed:.2f}s ({mode_label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
