"""DeepSeek OCR v2 (MLX/MPS) backend with in-process execution and CLI fallback."""

from .runner import run_for_files
from . import preflight

__all__ = ["run_for_files", "preflight"]
