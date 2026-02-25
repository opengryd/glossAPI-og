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

# Batch-size knobs: (env_var, config_section, config_key, default_for_mps)
# Reducing these from MinerU defaults (mfr: ~64, ocr-rec: ~6) prevents
# unified-memory thrashing on Apple Silicon, which causes throughput to
# collapse mid-stage (e.g. MFR Predict: 25 it/s → <3 it/s).
_BATCH_KNOBS = (
    # MFR (formula recognition) — most impactful; autoregressive model.
    ("GLOSSAPI_MINERU_MFR_BATCH_SIZE",     "formula-config", "mfr_batch_size",  32),
    # OCR recognition — text crop decoder.
    ("GLOSSAPI_MINERU_OCR_REC_BATCH_SIZE", "ocr-config",     "rec_batch_num",   6),
    # Layout detection — one-shot CNN, less sensitive but still tuneable.
    ("GLOSSAPI_MINERU_LAYOUT_BATCH_SIZE",  "layout-config",  "batch_size",      None),
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

    Batch-size env vars
    -------------------
    GLOSSAPI_MINERU_MFR_BATCH_SIZE
        Batch size for MFR (formula recognition).  Default applied when
        *device_mode* is ``"mps"``: 32.  Set to ``0`` to leave MinerU's
        own default untouched.
    GLOSSAPI_MINERU_OCR_REC_BATCH_SIZE
        Batch size for OCR recognition.  Default applied on MPS: 6.
    GLOSSAPI_MINERU_LAYOUT_BATCH_SIZE
        Batch size for layout detection.  No MPS default applied;
        set explicitly to override MinerU's default.
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

    # Apply batch-size knobs from env vars.
    # For MPS, inject defaults even when the env var is absent so that large
    # documents don't exhaust unified memory mid-stage.
    _is_mps = (device_mode == "mps") or (
        device_mode is None and data.get("device-mode") == "mps"
    )
    for env_key, section, field, mps_default in _BATCH_KNOBS:
        raw = (env or {}).get(env_key, "").strip()
        if raw:
            # Explicit override — always apply.
            try:
                val = int(raw)
            except ValueError:
                LOGGER.warning("%s=%r is not an integer; ignoring", env_key, raw)
                continue
            if val == 0:
                # Explicit 0 → leave MinerU's default untouched.
                continue
        elif _is_mps and mps_default is not None:
            # No explicit override, but we're on MPS and have a safe default.
            # Only inject if the config doesn't already set a value.
            existing = data.get(section, {})
            if isinstance(existing, dict) and field in existing:
                # Already set in magic-pdf.json — respect it.
                continue
            val = mps_default
        else:
            continue

        section_data = data.setdefault(section, {})
        if not isinstance(section_data, dict):
            LOGGER.warning(
                "Cannot inject %s=%d: config section %r is not a dict",
                field, val, section,
            )
            continue
        if section_data.get(field) != val:
            section_data[field] = val
            data[section] = section_data
            changed = True
            LOGGER.debug("Injecting %s.%s=%d (via %s)", section, field, val, env_key or "MPS default")

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
