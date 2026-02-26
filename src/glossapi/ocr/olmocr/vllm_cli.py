#!/usr/bin/env python3
"""OlmOCR-2 vLLM CLI — standalone script for in-process and subprocess CUDA inference.

This module can be run directly as a script
(``python -m glossapi.ocr.olmocr.vllm_cli``) or imported for in-process model
loading and PDF processing on NVIDIA GPUs via vLLM.

Heavy dependencies (vllm, torch, transformers, huggingface_hub) are imported
lazily so that importing the parent package remains lightweight.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

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

# The OlmOCR-2 prompt replicates the instruction template used during
# fine-tuning.  Keep this in sync with upstream ``olmocr.prompts`` and
# ``mlx_cli.py``.
DEFAULT_PROMPT = (
    "Below is the image of one page of a document. "
    "Your task is to convert the content of this page to markdown format.\n\n"
    "Requirements:\n"
    "- Convert all text content into proper markdown syntax\n"
    "- Preserve the document structure (headings, paragraphs, lists, etc.)\n"
    "- For tables, use markdown table syntax\n"
    "- For mathematical formulas, use LaTeX notation with $ delimiters\n"
    "- Preserve any special formatting (bold, italic, etc.)\n"
    "- Maintain the reading order\n"
    "- If the page is empty or contains no readable content, output [[Blank page]]\n\n"
    "Do not hallucinate. Output only what is in the image. "
    "Do not add any new information."
)

DEFAULT_DPI = 150
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"

# vLLM defaults
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


def _resolve_max_tokens(default: int = DEFAULT_MAX_TOKENS) -> int:
    """Return the effective max_tokens from env vars, with per-backend and global fallbacks.

    Priority: ``GLOSSAPI_OLMOCR_MAX_TOKENS`` > ``GLOSSAPI_VLM_MAX_TOKENS`` > *default*.
    """
    for var in ("GLOSSAPI_OLMOCR_MAX_TOKENS", "GLOSSAPI_VLM_MAX_TOKENS"):
        val = (os.getenv(var) or "").strip()
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return default


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    *,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> Any:
    """Load the vLLM model engine for OlmOCR-2 (Qwen2.5-VL).

    Returns an ``LLM`` instance ready for :func:`generate_page`.
    """
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )


def resolve_model_dir(model_dir: Optional[str] = None) -> str:
    """Resolve the CUDA model directory or HuggingFace identifier.

    Priority:
    1. ``GLOSSAPI_OLMOCR_MODEL_DIR`` environment variable
    2. *model_dir* argument (local path or HuggingFace repo id)
    3. ``GLOSSAPI_WEIGHTS_ROOT/olmocr/`` if present on disk
    4. HuggingFace Hub cache (already-downloaded snapshot)
    5. ``GLOSSAPI_OLMOCR_MODEL`` env var as HuggingFace repo id
    6. Default: ``allenai/olmOCR-2-7B-1025-FP8``

    Returns a string (local path or HuggingFace repo id) suitable for vLLM.
    """
    from glossapi.ocr.utils.weights import resolve_weights_dir

    log = logging.getLogger(__name__)

    # 1. Environment variable override
    env_dir = (os.getenv("GLOSSAPI_OLMOCR_MODEL_DIR") or "").strip()
    if env_dir:
        env_path = Path(env_dir).expanduser()
        if env_path.exists():
            return str(env_path)

    # 2. Explicit argument
    if model_dir:
        path = Path(model_dir).expanduser()
        if path.exists():
            return str(path)
        # Could be a HuggingFace repo id
        return str(model_dir)

    # 3. Weights root fallback
    resolved = resolve_weights_dir("olmocr", require_config_json=False)
    if resolved is not None:
        return str(resolved)

    # 4. HuggingFace Hub cache
    cached = _try_hf_cache(
        os.getenv("GLOSSAPI_OLMOCR_MODEL", "").strip() or DEFAULT_MODEL_ID
    )
    if cached is not None:
        log.info("OlmOCR vLLM model found in HF cache: %s", cached)
        return str(cached)

    # 5. Env var as model id, or default
    model_id = os.getenv("GLOSSAPI_OLMOCR_MODEL", "").strip() or DEFAULT_MODEL_ID
    return model_id


def _try_hf_cache(model_id: str) -> Optional[Path]:
    """Check whether *model_id* already exists in the HuggingFace Hub cache."""
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


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------


def render_page(doc_or_path: Any, page_index: int, dpi: int = DEFAULT_DPI) -> Any:
    """Render a single PDF page to a PIL Image via pypdfium2.

    *doc_or_path* may be either an open ``PdfDocument`` object **or** a
    ``Path`` / ``str`` pointing to the PDF file.  When a path is supplied a
    private document handle is created inside the call, which is required for
    thread-safe concurrent rendering — pypdfium2 document handles must not be
    shared across threads.
    """
    if isinstance(doc_or_path, (str, Path)):
        _doc = pdfium.PdfDocument(str(doc_or_path))
        page = _doc[page_index]
    else:
        page = doc_or_path[page_index]
    scale = float(dpi) / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def _image_to_base64(image: Any) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_page(
    llm: Any,
    image: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run inference on a single page image and return the markdown text.

    Uses vLLM's ``LLM.generate()`` with the OlmOCR-2 chat template.
    """
    from vllm import SamplingParams

    # Build the multi-modal chat message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template via the tokenizer
    tokenizer = llm.get_tokenizer()
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # vLLM multi-modal input
    outputs = llm.generate(
        [
            {
                "prompt": text_prompt,
                "multi_modal_data": {"image": image},
            }
        ],
        sampling_params=sampling_params,
    )

    if outputs and outputs[0].outputs:
        text = outputs[0].outputs[0].text.strip()
        return text if text else "[[Blank page]]"
    return "[[Blank page]]"


def generate_pages_batch(
    llm: Any,
    images: List[Any],
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[str]:
    """Process multiple page images in a single vLLM batched call.

    vLLM's continuous batching schedules all prompts efficiently in one pass,
    keeping the GPU fully utilised without idle cycles between pages.  This
    gives a 3–5× throughput improvement over one-page-at-a-time calls on
    multi-page documents.

    Returns a list of markdown strings in the same order as *images*.
    """
    from vllm import SamplingParams

    if not images:
        return []

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    batch: List[dict] = []
    for image in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch.append({"prompt": text_prompt, "multi_modal_data": {"image": image}})

    outputs = llm.generate(batch, sampling_params=sampling_params)

    results: List[str] = []
    for output in outputs:
        if output.outputs:
            text = output.outputs[0].text.strip()
            results.append(text if text else "[[Blank page]]")
        else:
            results.append("[[Blank page]]")
    return results


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
    llm: Any,
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

    _count_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(_count_doc)
    _count_doc.close()
    page_count = min(total_pages, max_pages) if max_pages else total_pages

    markdown_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_md = markdown_dir / f"{pdf_path.stem}.md"
    out_metrics = metrics_dir / f"{pdf_path.stem}.metrics.json"

    effective_max_tokens = _resolve_max_tokens() if max_tokens == DEFAULT_MAX_TOKENS else max_tokens
    lines: list[str] = []
    logger = logging.getLogger(__name__)

    # tqdm progress bar with logging fallback
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
    # Render all pages concurrently on the CPU while the GPU is idle.
    # Each render_page call receives the PDF *path* rather than a shared
    # PdfDocument handle.  pypdfium2 document objects are not thread-safe;
    # passing the path lets every worker thread open its own private handle.
    with ThreadPoolExecutor(max_workers=min(4, max(1, page_count))) as _render_pool:
        images = [
            f.result()
            for f in [_render_pool.submit(render_page, pdf_path, pi, dpi) for pi in range(page_count)]
        ]

    # Submit the entire document to vLLM as a single batched call.
    # vLLM continuous batching then schedules all prompts efficiently.
    texts = generate_pages_batch(llm, images, prompt=prompt, max_tokens=effective_max_tokens)

    for page_index, text in enumerate(texts):
        if content_debug:
            lines.append(f"<!-- page:{page_index + 1} -->")
        lines.append(text)
        lines.append("")
        if progress is not None:
            progress.update(1)
        else:
            logger.info(
                "OlmOCR vLLM: %s page %d/%d done",
                pdf_path.name,
                page_index + 1,
                page_count,
            )

    if progress is not None:
        progress.close()

    elapsed_total = time.time() - pdf_start
    logger.info(
        "OlmOCR vLLM: %s complete — %d pages in %.1fs (%.2fs/page)",
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
    """CLI entry-point for OlmOCR-2 vLLM PDF runner."""
    parser = argparse.ArgumentParser(description="OlmOCR-2 (vLLM/CUDA) PDF runner")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--content-debug", action="store_true")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Override the OCR prompt."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for vLLM KV-cache.",
    )

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    model_path = resolve_model_dir(args.model_dir)
    print(f"Loading OlmOCR-2 model via vLLM: {model_path}")
    llm = load_model(
        model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in input dir.")
        return 0

    start = time.time()
    for pdf_path in pdfs:
        process_pdf(
            pdf_path,
            output_dir,
            llm,
            prompt=args.prompt,
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
