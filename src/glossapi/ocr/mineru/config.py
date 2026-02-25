"""Auto-discover and resolve the MinerU ``magic-pdf.json`` configuration.

When ``MINERU_TOOLS_CONFIG_JSON`` is **not** set in the environment, MinerU
falls back to ``~/magic-pdf.json`` — which almost certainly does not exist
on a fresh clone.  This module transparently resolves the repo-bundled config
at ``model_weights/mineru/magic-pdf.json`` and rewrites relative paths
(``models-dir``, ``layoutreader-model-dir``) to absolute ones in a
temporary copy so MinerU picks everything up automatically.

Resolution order for the base config file
------------------------------------------
1. ``MINERU_TOOLS_CONFIG_JSON`` env var  (user / CI explicit override).
2. ``GLOSSAPI_WEIGHTS_ROOT/mineru/magic-pdf.json``  (unified weights root).
3. ``<repo_root>/model_weights/mineru/magic-pdf.json``  (convention default).

Path resolution inside the config
----------------------------------
Any value for ``models-dir`` or ``layoutreader-model-dir`` that is *not*
already an absolute path is resolved relative to the directory that contains
the config file.  A resolved copy is written to *tmp_root* (typically
``<output_dir>/mineru_tmp/``) so the original is never modified.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Keys whose values may be relative paths that need resolution.
_PATH_KEYS = ("models-dir", "layoutreader-model-dir")

# Enable/disable toggles: (env_var, config_section)
# Each env var maps to the ``enable`` key in the named config section.
# Set to ``0`` / ``false`` / ``no`` to skip the model pass entirely.
# Set to ``1`` / ``true`` / ``yes`` to force-enable.
_ENABLE_KNOBS = (
    ("GLOSSAPI_MINERU_FORMULA_ENABLE", "formula-config"),
    ("GLOSSAPI_MINERU_TABLE_ENABLE",   "table-config"),
)

# Well-known location of the config file beneath the weights root.
_MINERU_SUBPATH = Path("mineru") / "magic-pdf.json"


# ------------------------------------------------------------------
# Discover the config file
# ------------------------------------------------------------------

def _infer_repo_root() -> Optional[Path]:
    """Walk up from this file to find the repository root (has ``pyproject.toml``)."""
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return None


def discover_config(env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """Return the path to the best available ``magic-pdf.json``, or *None*.

    Parameters
    ----------
    env:
        Mapping to read environment variables from (defaults to ``os.environ``).
    """
    env = env if env is not None else dict(os.environ)

    # 1. Explicit env var
    explicit = (env.get("MINERU_TOOLS_CONFIG_JSON") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        LOGGER.debug("MINERU_TOOLS_CONFIG_JSON=%s does not exist", explicit)

    # 2. GLOSSAPI_WEIGHTS_ROOT/mineru/magic-pdf.json
    weights_root = (env.get("GLOSSAPI_WEIGHTS_ROOT") or "").strip()
    if weights_root:
        candidate = Path(weights_root) / _MINERU_SUBPATH
        if candidate.is_file():
            return candidate

    # 3. <repo_root>/model_weights/mineru/magic-pdf.json
    repo = _infer_repo_root()
    if repo is not None:
        candidate = repo / "model_weights" / _MINERU_SUBPATH
        if candidate.is_file():
            return candidate

    return None


# ------------------------------------------------------------------
# Resolve relative paths and write a temp copy
# ------------------------------------------------------------------

def _resolve_path(value: str, config_dir: Path) -> str:
    """If *value* is relative, make it absolute against *config_dir*."""
    p = Path(value)
    if p.is_absolute():
        return value
    resolved = (config_dir / p).resolve()
    return str(resolved)


def resolve_config(
    tmp_root: Path,
    *,
    env: Optional[Dict[str, str]] = None,
    device_mode: Optional[str] = None,
) -> Optional[Path]:
    """Discover, resolve, and (if needed) write a patched ``magic-pdf.json``.

    Returns the path to the config file that MinerU should use — either the
    original (if no rewriting was needed) or a temporary resolved copy beneath
    *tmp_root*.

    Parameters
    ----------
    tmp_root:
        Directory for writing the resolved temporary config.
    env:
        Environment mapping (defaults to ``os.environ``).
    device_mode:
        Optional device-mode override (``"cuda"``, ``"mps"``, ``"cpu"``).

    Enable/disable env vars
    -----------------------
    GLOSSAPI_MINERU_FORMULA_ENABLE
        Set to ``0`` to skip formula recognition (MFR) entirely.
        Set to ``1`` to force-enable regardless of the base config.
    GLOSSAPI_MINERU_TABLE_ENABLE
        Set to ``0`` to skip table extraction.
        Set to ``1`` to force-enable.
    """
    config_path = discover_config(env)
    if config_path is None:
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to parse MinerU config %s: %s", config_path, exc)
        return None

    config_dir = config_path.parent
    changed = False

    # Resolve relative path keys
    for key in _PATH_KEYS:
        val = data.get(key)
        if isinstance(val, str) and val:
            resolved = _resolve_path(val, config_dir)
            if resolved != val:
                data[key] = resolved
                changed = True
        elif isinstance(val, dict):
            # models-dir can be a dict mapping device → path
            for k, v in val.items():
                r = _resolve_path(v, config_dir)
                if r != v:
                    val[k] = r
                    changed = True

    # Auto-populate layoutreader-model-dir if missing
    if not data.get("layoutreader-model-dir"):
        models_dir = data.get("models-dir")
        if isinstance(models_dir, str):
            candidate = Path(models_dir) / "ReadingOrder" / "layout_reader"
            if candidate.is_dir():
                data["layoutreader-model-dir"] = str(candidate)
                changed = True

    # Apply device-mode override
    if device_mode and data.get("device-mode") != device_mode:
        data["device-mode"] = device_mode
        changed = True

    # Apply enable/disable toggles for optional model passes.
    for env_key, section in _ENABLE_KNOBS:
        raw = (env or {}).get(env_key, "").strip()
        if not raw:
            continue
        section_data = data.setdefault(section, {})
        if not isinstance(section_data, dict):
            continue
        enabled = raw.lower() not in {"0", "false", "no"}
        if section_data.get("enable") != enabled:
            section_data["enable"] = enabled
            data[section] = section_data
            changed = True
            LOGGER.debug("Setting %s.enable=%s (via %s)", section, enabled, env_key)

    if not changed:
        return config_path

    # Write resolved copy
    tmp_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{device_mode}" if device_mode else ""
    out_path = tmp_root / f"magic-pdf{suffix}.resolved.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("Wrote resolved MinerU config → %s", out_path)
    return out_path


# ------------------------------------------------------------------
# Convenience: prepare env dict with MINERU_TOOLS_CONFIG_JSON
# ------------------------------------------------------------------

def prepare_env_with_config(
    base_env: Dict[str, str],
    tmp_root: Path,
    *,
    device_mode: Optional[str] = None,
) -> Tuple[Dict[str, str], Optional[Path]]:
    """Return *(env, config_path)* with ``MINERU_TOOLS_CONFIG_JSON`` set.

    If the resolved config differs from what ``MINERU_TOOLS_CONFIG_JSON``
    already points to, a new env dict is returned with the updated value.
    Otherwise the original *base_env* is returned untouched.
    """
    resolved = resolve_config(tmp_root, env=base_env, device_mode=device_mode)
    if resolved is None:
        return base_env, None

    current = (base_env.get("MINERU_TOOLS_CONFIG_JSON") or "").strip()
    if current and Path(current).resolve() == resolved.resolve():
        return base_env, resolved

    env_copy = dict(base_env)
    env_copy["MINERU_TOOLS_CONFIG_JSON"] = str(resolved)
    return env_copy, resolved
