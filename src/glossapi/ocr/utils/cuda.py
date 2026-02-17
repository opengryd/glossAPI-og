"""Shared CUDA environment diagnostics for OCR backend runners.

Every CUDA-dependent backend (DeepSeek-OCR, OlmOCR, MinerU with CUDA, RapidOCR
with CUDA) can fail when the subprocess venv has a CPU-only PyTorch or CUDA
runtime libraries are not on the linker path.  This module provides reusable
helpers for:

* Detecting whether an exception is a CUDA environment problem.
* Collecting per-strategy errors and raising a clear diagnostic message.
* Pre-flight probing of a remote Python binary for ``torch.cuda.is_available()``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error-pattern matching
# ---------------------------------------------------------------------------

_CUDA_ERROR_PATTERNS = (
    "libcudart",
    "libcuda.so",
    "libnccl",
    "Torch not compiled with CUDA enabled",
    "CUDA driver version is insufficient",
    "no CUDA-capable device",
    "CUDA error",
    "cuda runtime error",
    "CUDA_HOME",
    "nvcc",
    "torch.cuda.is_available",
    "cannot open shared object file",
)


def is_cuda_setup_error(exc: Exception) -> bool:
    """Return *True* if *exc* looks like a CUDA environment problem."""
    text = str(exc).lower()
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if stderr:
        text = f"{text} {stderr.lower()}"
    return any(pat.lower() in text for pat in _CUDA_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Diagnostic error builder
# ---------------------------------------------------------------------------

# Maps backend name → (env var for Python binary, human-readable label).
_BACKEND_PYTHON_ENV: dict[str, Tuple[str, str]] = {
    "olmocr": ("GLOSSAPI_OLMOCR_PYTHON", "OlmOCR"),
    "deepseek-ocr": ("GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON", "DeepSeek-OCR"),
    "mineru": ("GLOSSAPI_MINERU_COMMAND", "MinerU"),
}


def raise_cuda_diagnosis(
    backend: str,
    strategy_errors: List[Tuple[str, Exception]],
    python_exe: Optional[Path] = None,
) -> None:
    """Raise a *RuntimeError* with CUDA-specific troubleshooting.

    Parameters
    ----------
    backend:
        Canonical backend name (``"olmocr"``, ``"deepseek-ocr"``, etc.).
    strategy_errors:
        List of ``(strategy_label, exception)`` pairs collected during the
        execution-strategy cascade.
    python_exe:
        The Python binary used for subprocess invocations.  When supplied,
        the diagnostic includes a verification command targeting it.
    """
    env_var, label = _BACKEND_PYTHON_ENV.get(backend, ("", backend))
    python_str = str(python_exe) if python_exe else "python"

    lines = [
        f"{label}: all CUDA execution strategies failed.  "
        f"The {label} subprocess environment does not have a working "
        "CUDA + PyTorch installation.",
        "",
        "Strategy failures:",
    ]
    for slabel, err in strategy_errors:
        lines.append(f"  - {slabel}: {err}")
    lines += [
        "",
        "Troubleshooting:",
        f"  1. Verify PyTorch CUDA in the {label} venv:",
        f"       {python_str} -c \"import torch; print(torch.cuda.is_available())\"",
        "  2. Install CUDA-enabled PyTorch if it reports False:",
        "       pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "  3. Ensure CUDA runtime libraries are findable:",
    ]
    if env_var and env_var != "GLOSSAPI_MINERU_COMMAND":
        ld_env = env_var.replace("_PYTHON", "_LD_LIBRARY_PATH").replace(
            "_TEST_PYTHON", "_LD_LIBRARY_PATH"
        )
        lines.append(f"       export {ld_env}=/usr/local/cuda/lib64")
    else:
        lines.append("       export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    lines += [
        f"  4. To use placeholder output instead, set GLOSSAPI_{backend.upper().replace('-', '_')}_ALLOW_STUB=1",
    ]
    raise RuntimeError("\n".join(lines)) from (
        strategy_errors[-1][1] if strategy_errors else None
    )


# ---------------------------------------------------------------------------
# Pre-flight CUDA probe
# ---------------------------------------------------------------------------

def probe_cuda(python_bin: Path, *, timeout: int = 30) -> Optional[bool]:
    """Probe a Python binary for ``torch.cuda.is_available()``.

    Returns
    -------
    True
        CUDA is available.
    False
        CUDA is explicitly not available (torch imported but GPU not found).
    None
        Could not determine (timeout, missing binary, import error, etc.).
    """
    if not python_bin.exists():
        return None
    try:
        result = subprocess.run(
            [str(python_bin), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    except Exception:
        return None

    if result.returncode != 0:
        # Could be torch not installed, or libcudart missing — treat as False.
        return False
    stdout = result.stdout.strip()
    if stdout == "True":
        return True
    if stdout == "False":
        return False
    return None
