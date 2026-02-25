from __future__ import annotations

import platform
from typing import Iterable, Optional


def _default_accel_type() -> str:
    """Return the best available accelerator type string for the current host.

    On macOS (Apple Silicon) the ``mps`` accelerator maps to Docling's
    ``AcceleratorDevice.MPS``, which routes ONNX execution through CoreML and
    surfaces the Neural Engine / Metal GPU.  On Linux/Windows we prefer CUDA
    when a GPU is present, otherwise AUTO lets Docling decide.
    """
    if platform.system() == "Darwin":
        # MPS is available on every Apple Silicon Mac; let Docling confirm via
        # torch.backends.mps.is_available() internally.
        return "mps"
    try:
        import torch  # type: ignore
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "CUDA"
    except Exception:
        pass
    return "AUTO"


def run_via_extract(
    corpus,
    files: Iterable[str],
    *,
    export_doc_json: bool = False,
    internal_debug: bool = False,
    content_debug: Optional[bool] = None,
) -> None:
    """Thin adapter that forwards to Corpus.extract for RapidOCR/Docling.

    This exists for symmetry with deepseek_runner and to keep the OCR package
    as the single entry point for OCR backends.
    """
    # Note: internal_debug/content_debug are no-ops for the Docling/RapidOCR path.
    # Docling's output already produces a single concatenated Markdown document.

    # Resolve the best accelerator for this host at call-time rather than
    # hard-coding "CUDA".  On macOS this resolves to "mps" so that the
    # downstream _resolve_accelerator() path maps ONNX sessions to
    # CoreMLExecutionProvider (ANE + Metal GPU) via SafeRapidOcrModel.
    accel = _default_accel_type()

    corpus.extract(
        input_format="pdf",
        num_threads=1,  # let extract decide; override in tests if needed
        accel_type=accel,
        force_ocr=True,
        formula_enrichment=False,
        code_enrichment=False,
        filenames=list(files),
        skip_existing=False,
        export_doc_json=bool(export_doc_json),
        emit_formula_index=bool(export_doc_json),
        phase1_backend="docling",
    )
