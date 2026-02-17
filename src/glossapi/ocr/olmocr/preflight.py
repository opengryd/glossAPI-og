"""Preflight checks for the OlmOCR-2 environment."""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    errors: List[CheckResult]
    warnings: List[CheckResult]
    infos: List[CheckResult]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summarize(self) -> str:
        lines: List[str] = []
        if self.errors:
            lines.append("Errors:")
            lines += [f"  - {c.name}: {c.message}" for c in self.errors]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {c.name}: {c.message}" for c in self.warnings]
        if self.infos:
            lines.append("Info:")
            lines += [f"  - {c.name}: {c.message}" for c in self.infos]
        return "\n".join(lines)


def _ensure_path(path: Path, label: str, errors: List[CheckResult]) -> Optional[Path]:
    if not path:
        errors.append(CheckResult(label, False, "Not provided"))
        return None
    if not path.exists():
        errors.append(CheckResult(label, False, f"Missing at {path}"))
        return None
    return path


def check_olmocr_env(
    env: Optional[Dict[str, str]] = None,
    *,
    check_gpu: bool = True,
) -> PreflightReport:
    """Validate OlmOCR-2 prerequisites without running OCR.

    Checks:
    - ``olmocr`` package importability (CUDA path)
    - ``poppler-utils`` availability (``pdftoppm`` on PATH)
    - CUDA GPU availability (optional, controlled by *check_gpu*)
    - MLX / ``mlx_vlm`` importability (macOS MPS path)
    - MPS device availability (macOS Apple Silicon)
    - Model path / identifier
    - ``GLOSSAPI_OLMOCR_ALLOW_CLI`` / ``GLOSSAPI_OLMOCR_ALLOW_STUB`` env flags
    """

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    # --- Stub / CLI flags ---
    allow_cli = env.get("GLOSSAPI_OLMOCR_ALLOW_CLI", "0") == "1"
    allow_stub = env.get("GLOSSAPI_OLMOCR_ALLOW_STUB", "1") == "1"
    if not allow_cli:
        warnings.append(
            CheckResult(
                "allow_cli",
                False,
                "Set GLOSSAPI_OLMOCR_ALLOW_CLI=1 to run the real OlmOCR pipeline.",
            )
        )
    if allow_stub:
        warnings.append(
            CheckResult(
                "allow_stub",
                False,
                "Set GLOSSAPI_OLMOCR_ALLOW_STUB=0 to fail instead of falling back to stub output.",
            )
        )

    # --- olmocr package ---
    try:
        import olmocr  # type: ignore

        version = getattr(olmocr, "__version__", "unknown")
        infos.append(CheckResult("olmocr", True, f"olmocr {version} import ok"))
    except Exception as exc:
        errors.append(CheckResult("olmocr", False, f"olmocr import failed: {exc}"))

    # --- poppler-utils (pdftoppm) ---
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        errors.append(
            CheckResult(
                "poppler",
                False,
                "pdftoppm not found on PATH; install poppler-utils "
                "(e.g. apt-get install poppler-utils or brew install poppler)",
            )
        )
    else:
        infos.append(CheckResult("poppler", True, f"pdftoppm found at {pdftoppm}"))

    # --- GPU availability ---
    if check_gpu:
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                infos.append(CheckResult("cuda", True, f"CUDA available — {gpu_name}"))
            else:
                errors.append(
                    CheckResult(
                        "cuda",
                        False,
                        "CUDA not available; OlmOCR-2 requires an NVIDIA GPU with ≥12 GB VRAM",
                    )
                )
        except Exception as exc:
            errors.append(CheckResult("cuda", False, f"torch import / CUDA check failed: {exc}"))

    # --- vLLM (CUDA in-process path) ---
    if not is_macos:
        try:
            import vllm  # type: ignore

            vllm_version = getattr(vllm, "__version__", "unknown")
            infos.append(
                CheckResult("vllm", True, f"vllm {vllm_version} import ok")
            )
        except Exception as exc:
            warnings.append(
                CheckResult(
                    "vllm",
                    False,
                    f"vllm not available — in-process CUDA execution disabled: {exc}",
                )
            )

    # --- CUDA model weights ---
    cuda_model_dir = env.get("GLOSSAPI_OLMOCR_MODEL_DIR", "").strip()
    if not cuda_model_dir:
        from glossapi.ocr.utils.weights import resolve_weights_dir as _resolve_cuda_wdir
        resolved_cuda = _resolve_cuda_wdir("olmocr", require_config_json=False)
        if resolved_cuda is not None:
            cuda_model_dir = str(resolved_cuda)
    if cuda_model_dir:
        cuda_md = Path(cuda_model_dir)
        if not cuda_md.exists():
            warnings.append(
                CheckResult(
                    "cuda_model_dir",
                    False,
                    f"CUDA model dir does not exist: {cuda_md}",
                )
            )
        else:
            has_config = (cuda_md / "config.json").exists()
            has_weights = any(cuda_md.glob("*.safetensors"))
            if not has_weights or not has_config:
                warnings.append(
                    CheckResult(
                        "cuda_model_contents",
                        False,
                        f"CUDA model dir {cuda_md} is missing weights and/or config.json",
                    )
                )
            else:
                infos.append(
                    CheckResult(
                        "cuda_model_dir",
                        True,
                        f"Local CUDA model dir: {cuda_md}",
                    )
                )

    # --- MLX / MPS (macOS Apple Silicon) ---
    is_macos = platform.system() == "Darwin"
    if is_macos:
        try:
            import mlx_vlm  # type: ignore

            mlx_version = getattr(mlx_vlm, "__version__", "unknown")
            infos.append(
                CheckResult("mlx_vlm", True, f"mlx_vlm {mlx_version} import ok")
            )
        except Exception as exc:
            warnings.append(
                CheckResult(
                    "mlx_vlm",
                    False,
                    f"mlx_vlm not available — in-process MLX execution disabled: {exc}",
                )
            )

        # Check MPS (Metal Performance Shaders) availability
        try:
            import torch as _torch_mps  # type: ignore

            if hasattr(_torch_mps.backends, "mps") and _torch_mps.backends.mps.is_available():
                infos.append(CheckResult("mps", True, "MPS backend available"))
            else:
                warnings.append(
                    CheckResult(
                        "mps",
                        False,
                        "MPS not available; Apple Silicon GPU acceleration disabled",
                    )
                )
        except Exception:
            # torch may not be installed — mlx_vlm doesn't require it
            try:
                import mlx.core as _mx  # type: ignore

                infos.append(
                    CheckResult(
                        "mps",
                        True,
                        "MLX metal backend available (torch not installed)",
                    )
                )
            except Exception:
                warnings.append(
                    CheckResult(
                        "mps",
                        False,
                        "Neither torch MPS nor MLX metal available",
                    )
                )

    # --- MLX model weights ---
    mlx_model_dir = env.get("GLOSSAPI_OLMOCR_MLX_MODEL_DIR", "").strip()
    mlx_model_id = env.get("GLOSSAPI_OLMOCR_MLX_MODEL", "").strip()
    # Fallback: check GLOSSAPI_WEIGHTS_ROOT/olmocr-mlx/
    if not mlx_model_dir:
        from glossapi.ocr.utils.weights import resolve_weights_dir
        resolved = resolve_weights_dir("olmocr-mlx", require_config_json=False)
        if resolved is not None:
            mlx_model_dir = str(resolved)
    if mlx_model_dir:
        md_mlx = Path(mlx_model_dir)
        if not md_mlx.exists():
            warnings.append(
                CheckResult(
                    "mlx_model_dir",
                    False,
                    f"MLX model dir does not exist: {md_mlx}",
                )
            )
        else:
            has_config = (md_mlx / "config.json").exists()
            has_weights = any(md_mlx.glob("*.safetensors"))
            if not has_weights or not has_config:
                warnings.append(
                    CheckResult(
                        "mlx_model_contents",
                        False,
                        f"MLX model dir {md_mlx} is missing weights and/or config.json",
                    )
                )
            else:
                infos.append(
                    CheckResult(
                        "mlx_model_dir",
                        True,
                        f"Local MLX model dir: {md_mlx}",
                    )
                )
    elif mlx_model_id:
        infos.append(
            CheckResult(
                "mlx_model",
                True,
                f"Using HuggingFace MLX model identifier: {mlx_model_id}",
            )
        )
    elif is_macos:
        infos.append(
            CheckResult(
                "mlx_model",
                True,
                "Using default HuggingFace MLX model: mlx-community/olmOCR-2-7B-1025-4bit",
            )
        )

    # --- Model ---
    model = env.get("GLOSSAPI_OLMOCR_MODEL", DEFAULT_MODEL)
    model_dir = env.get("GLOSSAPI_OLMOCR_MODEL_DIR", "").strip()
    if model_dir:
        md = _ensure_path(Path(model_dir), "model_dir", errors)
        if md:
            has_config = (md / "config.json").exists()
            has_weights = any(md.glob("*.safetensors"))
            if not has_weights or not has_config:
                errors.append(
                    CheckResult(
                        "model_contents",
                        False,
                        f"Model dir {md} is missing weights and/or config.json",
                    )
                )
            else:
                infos.append(CheckResult("model_dir", True, f"Local model dir: {md}"))
    else:
        infos.append(CheckResult("model", True, f"Using HuggingFace model identifier: {model}"))

    # --- Python binary ---
    python_bin = Path(env.get("GLOSSAPI_OLMOCR_PYTHON") or sys.executable)
    _ensure_path(python_bin, "olmocr_python", errors)

    # --- vLLM server (optional) ---
    server = env.get("GLOSSAPI_OLMOCR_SERVER", "").strip()
    if server:
        infos.append(CheckResult("server", True, f"External vLLM server: {server}"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_olmocr_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
