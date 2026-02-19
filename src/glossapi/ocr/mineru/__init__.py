"""MinerU OCR backend with a lightweight stub fallback."""

from . import config, preflight
from .runner import run_for_files

__all__ = ["config", "run_for_files", "preflight"]
