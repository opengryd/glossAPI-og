"""Centralized model-weight directory resolution via ``GLOSSAPI_WEIGHTS_ROOT``.

Every OCR backend that needs to locate local model weights should call
:func:`resolve_weights_dir` instead of reimplementing the look-up inline.

Resolution order
-----------------
1. *env_override* — a per-backend env var (e.g. ``GLOSSAPI_DEEPSEEK_MODEL_DIR``).
2. ``GLOSSAPI_WEIGHTS_ROOT/<subdir>`` — the unified weights-root convention.
3. ``None`` — caller decides the fallback (HF download, error, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_weights_dir(
    subdir: str,
    *,
    env_override: Optional[str] = None,
    require_config_json: bool = True,
) -> Optional[Path]:
    """Return the best local model-weight directory, or *None*.

    Parameters
    ----------
    subdir:
        Backend-specific subdirectory under ``GLOSSAPI_WEIGHTS_ROOT``
        (e.g. ``"deepseek-ocr"``, ``"olmocr-mlx"``).
    env_override:
        Value of a per-backend env var (already read by the caller).
        When non-empty and the path exists, it wins unconditionally.
    require_config_json:
        When *True* (default), the candidate directory must contain a
        ``config.json`` file to be accepted.

    Returns
    -------
    Path | None
        Resolved directory, or *None* if nothing valid was found.
    """
    # 1. Per-backend env override
    if env_override:
        override = env_override.strip()
        if override:
            p = Path(override).expanduser()
            if p.is_dir():
                return p

    # 2. GLOSSAPI_WEIGHTS_ROOT/<subdir>
    root = os.getenv("GLOSSAPI_WEIGHTS_ROOT", "").strip()
    if root:
        candidate = Path(root) / subdir
        if candidate.is_dir():
            if not require_config_json or (candidate / "config.json").exists():
                return candidate

    return None
