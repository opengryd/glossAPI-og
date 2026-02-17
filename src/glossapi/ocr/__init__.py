"""Lightweight OCR backend package.

Exports minimal, import-safe helpers for OCR backends. Heavy
dependencies (vLLM, transformers, PyMuPDF) are imported lazily
inside the specific backend functions so importing this package
does not require GPU stacks or model weights.
"""

from __future__ import annotations

import importlib

__all__ = [
    "deepseek",
    "deepseek_ocr2",
    "glm_ocr",
    "mineru",
    "olmocr",
    "rapidocr",
    "math",
    "utils",
    "deepseek_runner",
    "deepseek_ocr2_runner",
    "glm_ocr_runner",
    "mineru_runner",
    "olmocr_runner",
    "rapidocr_dispatch",
]

_SUBPACKAGES = {"deepseek", "deepseek_ocr2", "glm_ocr", "mineru", "olmocr", "rapidocr", "math", "utils"}
_ALIASES = {
    "deepseek_runner": "glossapi.ocr.deepseek.runner",
    "deepseek_ocr2_runner": "glossapi.ocr.deepseek_ocr2.runner",
    "glm_ocr_runner": "glossapi.ocr.glm_ocr.runner",
    "mineru_runner": "glossapi.ocr.mineru.runner",
    "olmocr_runner": "glossapi.ocr.olmocr.runner",
    "rapidocr_dispatch": "glossapi.ocr.rapidocr.dispatch",
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"glossapi.ocr.{name}")
    target = _ALIASES.get(name)
    if target:
        return importlib.import_module(target)
    raise AttributeError(name)
