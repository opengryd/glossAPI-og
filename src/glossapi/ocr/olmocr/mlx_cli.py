#!/usr/bin/env python3
"""OlmOCR-2 MLX CLI — standalone script for in-process and subprocess invocation.

This module can be run directly as a script
(``python -m glossapi.ocr.olmocr.mlx_cli``) or imported for in-process model
loading and PDF processing on Apple Silicon via MLX.

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

# The OlmOCR-2 prompt replicates the instruction template used during
# fine-tuning.  Keep this in sync with upstream ``olmocr.prompts``.
DEFAULT_PROMPT = (
    "<image>\nBelow is the image of one page of a document. "
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
DEFAULT_MODEL_ID = "mlx-community/olmOCR-2-7B-1025-4bit"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_model_and_processor(model_path: Path) -> tuple:
    """Load the MLX model and processor for OlmOCR-2 (Qwen2.5-VL).

    Returns ``(model, processor)`` ready for :func:`generate_page`.

    Qwen2.5-VL is natively supported by ``mlx-vlm``, so loading is simpler
    than DeepSeek OCR v2's custom processor path.  A fallback is included for
    tokenizer compatibility issues.
    """
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


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """Resolve the model directory: env var > explicit arg > weights root > HuggingFace download.

    Priority:
    1. ``GLOSSAPI_OLMOCR_MLX_MODEL_DIR`` environment variable
    2. *model_dir* argument (local path or HuggingFace repo id)
    3. ``GLOSSAPI_WEIGHTS_ROOT/olmocr-mlx/`` if present on disk
    4. ``GLOSSAPI_OLMOCR_MLX_MODEL`` env var as HuggingFace repo id
    5. Auto-download from ``mlx-community/olmOCR-2-7B-1025-4bit``
    """
    from huggingface_hub import snapshot_download
    from glossapi.ocr.utils.weights import resolve_weights_dir

    env_dir = (os.getenv("GLOSSAPI_OLMOCR_MLX_MODEL_DIR") or "").strip()
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
        resolved = resolve_weights_dir("olmocr-mlx")
        if resolved is not None:
            return resolved
        model_id = (
            os.getenv("GLOSSAPI_OLMOCR_MLX_MODEL", "").strip() or DEFAULT_MODEL_ID
        )

    print(f"Downloading model: {model_id}")
    local_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.py",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ],
    )
    return Path(local_dir)


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------


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
    """Run inference on a single page image and return the markdown text."""
    from mlx_vlm import generate

    result = generate(
        model,
        processor,
        prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=0.0,
        skip_special_tokens=True,
    )
    # mlx_vlm.generate may return a string or an object with a .text attribute
    if isinstance(result, str):
        text = result.strip()
    else:
        text = (getattr(result, "text", None) or str(result)).strip()
    return text if text else "[[Blank page]]"


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
    for page_index in range(page_count):
        if content_debug:
            lines.append(f"<!-- page:{page_index + 1} -->")
        image = render_page(doc, page_index, dpi)
        text = generate_page(
            model, processor, image, prompt=prompt, max_tokens=max_tokens
        )
        lines.append(text)
        lines.append("")
        if progress is not None:
            progress.update(1)
        else:
            logger.info(
                "OlmOCR MLX: %s page %d/%d done",
                pdf_path.name,
                page_index + 1,
                page_count,
            )

    if progress is not None:
        progress.close()

    elapsed_total = time.time() - pdf_start
    logger.info(
        "OlmOCR MLX: %s complete — %d pages in %.1fs (%.2fs/page)",
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
    """CLI entry-point for OlmOCR-2 MLX PDF runner."""
    parser = argparse.ArgumentParser(description="OlmOCR-2 (MLX) PDF runner")
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
        "--prompt", default=DEFAULT_PROMPT, help="Override the OCR prompt."
    )

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

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
