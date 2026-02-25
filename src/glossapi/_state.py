"""Shared backend-state helpers for the GlossAPI CLI.

``glossapi setup`` writes to the state file after a successful install;
``glossapi pipeline`` and ``glossapi status`` read from it so the two
commands share a common understanding of which backends are available and
where their virtual environments live.

The file lives at ``dependency_setup/.glossapi_state.json`` (resolved
relative to the current working directory, which is the project root when
the CLI is used normally).  It is gitignored and may contain local paths.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

STATE_FILE = Path("dependency_setup") / ".glossapi_state.json"

_ALL_BACKENDS = [
    "vanilla",
    "rapidocr",
    "mineru",
    "deepseek-ocr",
    "deepseek-ocr-2",
    "glm-ocr",
    "olmocr",
]

# ---------------------------------------------------------------------------
# Internal I/O — atomic read / write
# ---------------------------------------------------------------------------


def _load() -> Dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, STATE_FILE)


# ---------------------------------------------------------------------------
# Write API — called by setup_wizard after a successful installation
# ---------------------------------------------------------------------------


def record_setup(
    mode: str,
    venv: Path,
    *,
    python_bin: str,
    weights_downloaded: Optional[List[str]] = None,
) -> None:
    """Record a successful backend installation in the state file.

    Parameters
    ----------
    mode:
        Backend profile name (e.g. ``"rapidocr"``, ``"deepseek-ocr"``).
    venv:
        Path to the virtual environment that was created.
    python_bin:
        Python executable that was used to create the venv.
    weights_downloaded:
        Optional list of backend names whose model weights were also
        downloaded during this setup run.
    """
    data = _load()
    backends: Dict[str, Any] = data.setdefault("backends", {})
    backends[mode] = {
        "installed": True,
        "venv": str(venv.resolve()),
        "python": python_bin,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    if weights_downloaded:
        weights: Dict[str, bool] = data.setdefault("weights", {})
        for w in weights_downloaded:
            weights[w] = True
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save(data)


# ---------------------------------------------------------------------------
# Read API — called by pipeline_wizard and the status command
# ---------------------------------------------------------------------------


def get_installed_backends() -> Dict[str, Dict[str, Any]]:
    """Return ``{mode: info_dict}`` for every backend recorded as installed."""
    return _load().get("backends", {})


def is_backend_installed(mode: str) -> bool:
    """Return ``True`` if the given backend has been successfully set up."""
    return get_installed_backends().get(mode, {}).get("installed", False)


def get_backend_venv(mode: str) -> Optional[Path]:
    """Return the venv ``Path`` recorded for *mode*, or ``None``."""
    v = get_installed_backends().get(mode, {}).get("venv")
    return Path(v) if v else None


def get_weights_info() -> Dict[str, bool]:
    """Return ``{backend: True}`` for every backend whose weights are recorded."""
    return _load().get("weights", {})


def all_state() -> Dict[str, Any]:
    """Return the raw state dict (empty dict if no file exists)."""
    return _load()


def has_state_file() -> bool:
    """Return ``True`` if a state file already exists on disk."""
    return STATE_FILE.exists()


# ---------------------------------------------------------------------------
# Pipeline state — lightweight key/value store inside the same JSON file
# ---------------------------------------------------------------------------


def get_pipeline_state(key: str) -> Optional[str]:
    """Return a pipeline-state value by *key*, or ``None`` if absent."""
    return _load().get("pipeline", {}).get(key)


def set_pipeline_state(key: str, value: str) -> None:
    """Persist a pipeline-state *key*/*value* pair in the state file."""
    data = _load()
    data.setdefault("pipeline", {})[key] = value
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save(data)
