"""Preflight checks for the DeepSeek OCR CLI environment."""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from glossapi.ocr.utils.weights import resolve_weights_dir

DEFAULT_SCRIPT = Path.cwd() / "deepseek-ocr" / "run_pdf_ocr_vllm.py"
DEFAULT_LIB_DIR = Path.cwd() / "deepseek-ocr" / "libjpeg-turbo" / "lib"

# Embedded MLX CLI script shipped with the package.
_PACKAGE_MLX_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


def _has_mlx_weights(model_dir: Path) -> bool:
    patterns = ["*.safetensors", "*.npz", "*.bin", "*.mlx", "*.pt"]
    for pattern in patterns:
        if any(model_dir.glob(pattern)):
            return True
    return False


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


def check_deepseek_ocr_env(
    env: Optional[Dict[str, str]] = None,
    *,
    check_flashinfer: bool = True,
) -> PreflightReport:
    """Validate DeepSeek OCR CLI prerequisites without running the model.

    On macOS the MPS/MLX path is checked (mlx-vlm, Metal availability, MLX
    model weights).  On Linux/Windows the CUDA/vLLM path is checked (vLLM
    script, cc1plus, flashinfer, CUDA model weights, ld_library_path).
    """

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    enable_ocr = env.get("GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR", "0") == "1"
    enable_stub = env.get("GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB", "1") == "1"
    if not enable_ocr:
        warnings.append(
            CheckResult(
                "enable_ocr",
                False,
                "Set GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR=1 to force the real CLI.",
            )
        )
    if enable_stub:
        warnings.append(
            CheckResult(
                "enable_stub",
                False,
                "Set GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0 to fail instead of falling back to stub output.",
            )
        )

    # ----- Device detection -----
    is_macos = platform.system() == "Darwin"
    env_device = env.get("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "").strip().lower()
    if env_device:
        active_device = env_device
    else:
        active_device = "mps" if is_macos else "cuda"

    infos.append(
        CheckResult("device", True, f"Active device: {active_device}")
    )

    # =========================================================================
    # macOS / MPS path
    # =========================================================================
    if active_device == "mps" or (is_macos and active_device not in ("cuda",)):
        # mlx-vlm availability
        try:
            import mlx_vlm  # type: ignore

            mlx_version = getattr(mlx_vlm, "__version__", "unknown")
            infos.append(CheckResult("mlx_vlm", True, f"mlx-vlm {mlx_version} import ok"))
        except Exception as exc:  # pragma: no cover - depends on env
            warnings.append(
                CheckResult(
                    "mlx_vlm",
                    False,
                    f"mlx-vlm not available — in-process MLX execution disabled: {exc}",
                )
            )

        # Metal / MPS availability
        try:
            import torch as _torch_mps  # type: ignore

            if hasattr(_torch_mps.backends, "mps") and _torch_mps.backends.mps.is_available():
                infos.append(CheckResult("mps", True, "MPS backend available"))
            else:
                warnings.append(
                    CheckResult("mps", False, "torch MPS not available; Apple Silicon GPU acceleration disabled")
                )
        except Exception:
            # torch not installed; check via MLX directly
            try:
                import mlx.core as _mx  # type: ignore

                infos.append(CheckResult("mps", True, "MLX Metal backend available (torch not installed)"))
            except Exception:
                warnings.append(
                    CheckResult("mps", False, "Neither torch MPS nor MLX Metal is available")
                )

        # MLX CLI script
        mlx_script_env = env.get("GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT", "").strip()
        mlx_script = Path(mlx_script_env) if mlx_script_env else _PACKAGE_MLX_CLI_SCRIPT
        if mlx_script.exists():
            infos.append(CheckResult("mlx_script", True, f"MLX script: {mlx_script}"))
        else:
            warnings.append(
                CheckResult("mlx_script", False, f"MLX script not found at {mlx_script}")
            )

        # MLX model weights
        mlx_model_dir_env = (
            env.get("GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR") or ""
        ).strip()
        mlx_model_dir: Optional[Path] = None
        if mlx_model_dir_env:
            mlx_model_dir = _ensure_path(Path(mlx_model_dir_env), "mlx_model_dir", errors)
        else:
            resolved_mlx = resolve_weights_dir("deepseek-ocr-1-mlx", require_config_json=False)
            if resolved_mlx is not None:
                mlx_model_dir = resolved_mlx
            else:
                infos.append(
                    CheckResult(
                        "mlx_model_dir",
                        True,
                        "No MLX model dir configured; will auto-download from HuggingFace at runtime "
                        "(set GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR or GLOSSAPI_WEIGHTS_ROOT to avoid this).",
                    )
                )
        if mlx_model_dir:
            has_config = (mlx_model_dir / "config.json").exists()
            has_weights = _has_mlx_weights(mlx_model_dir)
            if not has_weights or not has_config:
                errors.append(
                    CheckResult(
                        "mlx_model_contents",
                        False,
                        f"MLX model dir {mlx_model_dir} is missing weights and/or config.json",
                    )
                )
            else:
                infos.append(CheckResult("mlx_model_dir", True, f"Local MLX model dir: {mlx_model_dir}"))

    # =========================================================================
    # CUDA / vLLM path (Linux / Windows)
    # =========================================================================
    else:
        script = Path(env.get("GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT") or DEFAULT_SCRIPT)
        _ensure_path(script, "vllm_script", errors)

        python_bin = Path(env.get("GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON") or sys.executable)
        _ensure_path(python_bin, "deepseek_ocr_python", errors)

        raw_model_dir = (
            env.get("GLOSSAPI_DEEPSEEK_OCR_TEST_MODEL_DIR")
            or env.get("GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR")
            or ""
        ).strip()
        if not raw_model_dir:
            resolved = resolve_weights_dir("deepseek-ocr", require_config_json=False)
            if resolved:
                raw_model_dir = str(resolved)
        if raw_model_dir:
            model_dir = _ensure_path(Path(raw_model_dir), "model_dir", errors)
        else:
            model_dir = None
            errors.append(
                CheckResult(
                    "model_dir",
                    False,
                    "No GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR or GLOSSAPI_WEIGHTS_ROOT configured.",
                )
            )
        if model_dir:
            has_weights = (
                any(model_dir.glob("*.safetensors"))
                or (model_dir / "model-00001-of-000001.safetensors").exists()
            )
            has_config = (model_dir / "config.json").exists()
            if not has_weights or not has_config:
                errors.append(
                    CheckResult(
                        "model_contents",
                        False,
                        f"Model dir {model_dir} is missing weights/config.json",
                    )
                )

        ld_path_env = env.get("GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH")
        lib_dir = Path(ld_path_env) if ld_path_env else DEFAULT_LIB_DIR
        _ensure_path(lib_dir, "ld_library_path", errors)

        cc1plus_path = shutil.which("cc1plus", path=env.get("PATH", ""))
        if not cc1plus_path:
            errors.append(
                CheckResult(
                    "cc1plus",
                    False,
                    "C++ toolchain missing (cc1plus not on PATH); install g++ and ensure PATH includes gcc's cc1plus.",
                )
            )
        else:
            infos.append(CheckResult("cc1plus", True, f"Found at {cc1plus_path}"))

        if check_flashinfer:
            try:
                import flashinfer  # type: ignore

                infos.append(CheckResult("flashinfer", True, f"flashinfer {flashinfer.__version__} import ok"))
            except Exception as exc:  # pragma: no cover - depends on env
                errors.append(CheckResult("flashinfer", False, f"flashinfer import failed: {exc}"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_deepseek_ocr_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
