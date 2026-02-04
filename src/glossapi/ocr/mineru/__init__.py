"""MinerU OCR backend with a lightweight stub fallback."""

from . import preflight
from .runner import run_for_files

__all__ = ["run_for_files", "preflight"]
