#!/usr/bin/env python3
"""DeepSeek OCR v2 MLX CLI — standalone script for subprocess invocation.

This module can be run directly as a script (``python -m glossapi.ocr.deepseek_ocr2.mlx_cli``)
or imported for in-process model loading and PDF processing.

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
from collections import deque
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

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_DPI = 150  # 150 DPI keeps A4 longest side ≤ ~1240px, matching the model's 1024px tile budget
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL_ID = "mlx-community/DeepSeek-OCR-2-8bit"


def _resolve_max_tokens(default: int = DEFAULT_MAX_TOKENS) -> int:
    """Return the effective max_tokens from env vars, with per-backend and global fallbacks.

    Priority: ``GLOSSAPI_DEEPSEEK2_MAX_TOKENS`` > ``GLOSSAPI_VLM_MAX_TOKENS`` > *default*.
    """
    for var in ("GLOSSAPI_DEEPSEEK2_MAX_TOKENS", "GLOSSAPI_VLM_MAX_TOKENS"):
        val = (os.getenv(var) or "").strip()
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return default


def _render_prefetch_depth() -> int:
    """Return the configured render pre-fetch depth (1–4, default 2).

    Controlled by ``GLOSSAPI_VLM_RENDER_PREFETCH``.  A depth of 2 keeps two
    page renders in flight concurrently with GPU inference, hiding the
    ~50–150 ms rasterisation latency on large pages.
    """
    try:
        return max(1, min(4, int(os.getenv("GLOSSAPI_VLM_RENDER_PREFETCH", "2"))))
    except (ValueError, TypeError):
        return 2


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_processor_config(model_path: Path) -> dict:
    config_path = model_path / "processor_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _build_deepseek_processor(tokenizer: Any, model_path: Path) -> Any:
    from mlx_vlm.models.deepseekocr_2.processing_deepseekocr import DeepseekOCR2Processor

    cfg = _load_processor_config(model_path)
    candidate_resolutions = cfg.get("candidate_resolutions") or [(1024, 1024)]
    candidate_resolutions = tuple(tuple(r) for r in candidate_resolutions)
    image_mean = tuple(cfg.get("image_mean") or (0.5, 0.5, 0.5))
    image_std = tuple(cfg.get("image_std") or (0.5, 0.5, 0.5))

    return DeepseekOCR2Processor(
        tokenizer=tokenizer,
        candidate_resolutions=candidate_resolutions,
        patch_size=cfg.get("patch_size", 16),
        downsample_ratio=cfg.get("downsample_ratio", 4),
        image_mean=image_mean,
        image_std=image_std,
        normalize=cfg.get("normalize", True),
        image_token=cfg.get("image_token", "<image>"),
        pad_token=cfg.get("pad_token", "<|pad|>"),
        add_special_token=cfg.get("add_special_token", False),
        sft_format=cfg.get("sft_format", "deepseek"),
        mask_prompt=cfg.get("mask_prompt", True),
        ignore_id=cfg.get("ignore_id", -100),
    )


def _attach_detokenizer(processor: Any, tokenizer: Any) -> None:
    from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from mlx_vlm.utils import StoppingCriteria

    processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        eos_token_ids = [eos_token_id] if eos_token_id is not None else []
    criteria = StoppingCriteria(eos_token_ids, tokenizer)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.stopping_criteria = criteria
    else:
        processor.stopping_criteria = criteria


def load_model_and_processor(model_path: Path) -> tuple:
    """Load the MLX model and processor, with fallback for tokenizer issues.

    Sets the Metal wired memory limit before weight allocation to prevent
    unified-memory page-outs, then compiles Metal shaders with a dummy
    inference so the first real page does not pay the JIT cost.

    Returns ``(model, processor)`` ready for :func:`generate_page`.
    """
    _set_metal_wired_limit()
    model, processor = _load_impl(model_path)
    _warmup_metal_shaders(model, processor)
    return model, processor


def _load_impl(model_path: Path) -> tuple:
    """Internal: raw model + processor load with tokenizer-class fallback."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_model as _load_model

    try:
        return load(str(model_path), trust_remote_code=True)
    except ValueError as exc:
        if "Unrecognized processing class" not in str(exc):
            raise

        from transformers import AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        except ValueError as tok_exc:
            if "TokenizersBackend" not in str(tok_exc):
                raise
            tokenizer_file = model_path / "tokenizer.json"
            tokenizer_config = model_path / "tokenizer_config.json"
            if not tokenizer_file.exists():
                raise RuntimeError(
                    "DeepSeek OCR v2 tokenizer.json missing; model repo is incomplete."
                ) from tok_exc
            extra_special_tokens: list = []
            bos_token = eos_token = pad_token = unk_token = None
            if tokenizer_config.exists():
                config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
                extra_special_tokens = config.get("extra_special_tokens") or []
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                pad_token = config.get("pad_token")
                unk_token = config.get("unk_token")
            try:
                tokenizer = LlamaTokenizerFast(
                    tokenizer_file=str(tokenizer_file),
                    bos_token=bos_token,
                    eos_token=eos_token,
                    pad_token=pad_token,
                    unk_token=unk_token,
                    additional_special_tokens=extra_special_tokens,
                )
            except Exception:
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_file),
                    bos_token=bos_token,
                    eos_token=eos_token,
                    pad_token=pad_token,
                    unk_token=unk_token,
                    additional_special_tokens=extra_special_tokens,
                )

        model = _load_model(model_path, trust_remote_code=True)
        processor = _build_deepseek_processor(tokenizer, model_path)
        _attach_detokenizer(processor, tokenizer)
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
                revisions = sorted(
                    repo.revisions,
                    key=lambda r: r.last_modified,
                    reverse=True,
                )
                if revisions:
                    snap = revisions[0].snapshot_path
                    if any(snap.glob("*.safetensors")):
                        return snap
    except Exception:
        pass
    return None


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """Resolve the model directory: env var > explicit arg > weights root > HF cache > download.

    Priority:
    1. ``GLOSSAPI_DEEPSEEK2_MODEL_DIR`` environment variable
    2. *model_dir* argument (local path or HuggingFace repo id)
    3. ``GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-mlx/`` if present on disk
    4. HuggingFace Hub cache (already-downloaded snapshot)
    5. Auto-download from ``mlx-community/DeepSeek-OCR-2-8bit``

    When downloading, if ``GLOSSAPI_WEIGHTS_ROOT`` is set the model is saved
    directly into ``<root>/deepseek-ocr-mlx/`` so that subsequent runs resolve
    at step 3 without network access.
    """
    from huggingface_hub import snapshot_download
    from glossapi.ocr.utils.weights import resolve_weights_dir

    log = logging.getLogger(__name__)

    env_dir = (os.getenv("GLOSSAPI_DEEPSEEK2_MODEL_DIR") or "").strip()
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
        resolved = resolve_weights_dir("deepseek-ocr-mlx")
        if resolved is not None:
            return resolved
        model_id = DEFAULT_MODEL_ID

    # Check HuggingFace Hub cache before triggering a download
    cached = _try_hf_cache(model_id)
    if cached is not None:
        log.info("DeepSeek OCR v2 MLX model found in HF cache: %s", cached)
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
        local_target = weights_root / "deepseek-ocr-mlx"
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


def render_page(
    doc_or_path: Any,
    page_index: int,
    dpi: int = DEFAULT_DPI,
    max_side: int = 1344,
) -> Any:
    """Render a single PDF page to a PIL Image via pypdfium2.

    *doc_or_path* may be either an open ``PdfDocument`` object **or** a
    ``Path`` / ``str`` pointing to the PDF file.  When a path is supplied a
    private document handle is created inside the call, which is required for
    thread-safe concurrent rendering — pypdfium2 document handles must not be
    shared across threads.

    *max_side* caps the longest edge of the rendered image before it is passed
    to the VLM processor.  DeepSeek-OCR tiles images at 1024 px; keeping at or
    below 1344 px avoids an over-sized render while preserving quality within
    the model's effective resolution range.
    """
    if isinstance(doc_or_path, (str, Path)):
        _doc = pdfium.PdfDocument(str(doc_or_path))
        page = _doc[page_index]
    else:
        page = doc_or_path[page_index]
    scale = float(dpi) / 72.0
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    w, h = image.size
    if max(w, h) > max_side:
        factor = max_side / max(w, h)
        image = image.resize(
            (max(1, int(w * factor)), max(1, int(h * factor))),
            resample=Image.LANCZOS,
        )
    return image


def generate_page(
    model: Any,
    processor: Any,
    image: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run inference on a single page image and return the markdown text."""
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
    # mlx_vlm.generate may return a plain str (older versions) or a
    # GenerationOutput object with a .text attribute (newer versions).
    if isinstance(result, str):
        text = result.strip()
    else:
        text = (getattr(result, "text", None) or "").strip()
    return text if text else "[[Blank page]]"


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
) -> int:
    """Process a single PDF and write markdown + metrics files.

    Returns the number of pages processed.
    """
    if pdfium is None:
        raise RuntimeError(f"pypdfium2 is required but unavailable: {_PDFIUM_ERROR}")
    if Image is None:
        raise RuntimeError(f"Pillow is required but unavailable: {_PIL_ERROR}")

    with pdfium.PdfDocument(str(pdf_path)) as _count_doc:
        total_pages = len(_count_doc)
    page_count = min(total_pages, max_pages) if max_pages else total_pages

    markdown_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_md = markdown_dir / f"{pdf_path.stem}.md"
    out_metrics = metrics_dir / f"{pdf_path.stem}.metrics.json"

    effective_max_tokens = _resolve_max_tokens() if max_tokens == DEFAULT_MAX_TOKENS else max_tokens
    _prefetch = _render_prefetch_depth()
    lines: list[str] = []
    logger = logging.getLogger(__name__)

    # tqdm progress bar (matches RapidOCR style), with logging fallback
    try:
        from tqdm import tqdm as _tqdm
        progress = _tqdm(
            total=page_count,
            desc=f"OCR {pdf_path.name}",
            unit="page",
            dynamic_ncols=True,
        )
    except ImportError:
        progress = None

    pdf_start = time.time()
    # Each render_page call receives the PDF *path* rather than a shared
    # PdfDocument handle.  pypdfium2 document objects are not thread-safe;
    # passing the path lets every worker thread open its own private handle,
    # preventing the segfault that occurred when the document was shared.
    with ThreadPoolExecutor(max_workers=_prefetch) as _render_pool:
        # Pre-fill the prefetch queue with the first _prefetch pages.
        _render_q: deque = deque(
            _render_pool.submit(render_page, pdf_path, pi, dpi)
            for pi in range(min(_prefetch, page_count))
        )
        for page_index in range(page_count):
            if content_debug:
                lines.append(f"<!-- page:{page_index + 1} -->")
            # Enqueue the next page not yet in the queue.
            next_pi = page_index + _prefetch
            if next_pi < page_count:
                _render_q.append(_render_pool.submit(render_page, pdf_path, next_pi, dpi))
            image = _render_q.popleft().result()
            text = generate_page(model, processor, image, prompt=prompt, max_tokens=effective_max_tokens)
            _flush_metal()  # flush Metal command buffer and release activation slabs
            lines.append(text)
            lines.append("")
            if progress is not None:
                progress.update(1)
            else:
                logger.info(
                    "DeepSeek OCR v2: %s page %d/%d done",
                    pdf_path.name, page_index + 1, page_count,
                )

    if progress is not None:
        progress.close()

    elapsed_total = time.time() - pdf_start
    logger.info(
        "DeepSeek OCR v2: %s complete — %d pages in %.1fs (%.2fs/page)",
        pdf_path.name, page_count, elapsed_total,
        elapsed_total / max(page_count, 1),
    )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    _write_metrics(out_metrics, page_count)
    return page_count


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek OCR v2 (MLX) PDF runner")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--content-debug", action="store_true")
    parser.add_argument("--device", default=None, help="Optional device hint (unused).")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    prompt = DEFAULT_PROMPT

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    model_path = resolve_model_dir(args.model_dir)
    model, processor = load_model_and_processor(model_path)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in input dir.")
        return 0

    start = time.time()
    for pdf_path in pdfs:
        process_pdf(
            pdf_path,
            output_dir,
            model,
            processor,
            prompt=prompt,
            max_pages=args.max_pages,
            dpi=args.dpi,
            max_tokens=args.max_tokens,
            content_debug=bool(args.content_debug),
        )
    elapsed = time.time() - start
    print(f"Processed {len(pdfs)} PDF(s) in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
