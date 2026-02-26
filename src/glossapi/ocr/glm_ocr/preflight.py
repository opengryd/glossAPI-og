"""Preflight checks for the GLM-OCR environment."""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


def check_glmocr_env(
    env: Optional[Dict[str, str]] = None,
) -> PreflightReport:
    """Validate GLM-OCR prerequisites without running OCR.

    Checks:
    - MLX / ``mlx_vlm`` importability (macOS MPS path)
    - MPS device availability (macOS Apple Silicon)
    - Model path / identifier
    - ``GLOSSAPI_GLMOCR_ENABLE_OCR`` / ``GLOSSAPI_GLMOCR_ENABLE_STUB`` env flags
    """

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    # --- Stub / CLI flags ---
    enable_ocr = env.get("GLOSSAPI_GLMOCR_ENABLE_OCR", "0") == "1"
    enable_stub = env.get("GLOSSAPI_GLMOCR_ENABLE_STUB", "1") == "1"
    if not enable_ocr:
        warnings.append(
            CheckResult(
                "enable_ocr",
                False,
                "Set GLOSSAPI_GLMOCR_ENABLE_OCR=1 to run the real GLM-OCR pipeline.",
            )
        )
    if enable_stub:
        warnings.append(
            CheckResult(
                "enable_stub",
                False,
                "Set GLOSSAPI_GLMOCR_ENABLE_STUB=0 to fail instead of falling back to stub output.",
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
            # glm_ocr model type requires mlx-vlm >= 0.3.12
            _min_ver = "0.3.12"
            try:
                from packaging.version import Version
                if mlx_version != "unknown" and Version(mlx_version) < Version(_min_ver):
                    errors.append(
                        CheckResult(
                            "mlx_vlm_version",
                            False,
                            f"mlx-vlm>={_min_ver} required for glm_ocr model type, "
                            f"but {mlx_version} is installed. Run: pip install --upgrade mlx-vlm",
                        )
                    )
                else:
                    infos.append(
                        CheckResult("mlx_vlm_version", True, f"mlx-vlm {mlx_version} >= {_min_ver}")
                    )
            except Exception:
                warnings.append(
                    CheckResult(
                        "mlx_vlm_version",
                        False,
                        f"Could not verify mlx-vlm version ({mlx_version}); "
                        f">={_min_ver} required for glm_ocr model type",
                    )
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
            import transformers as _tf  # type: ignore

            tf_version = getattr(_tf, "__version__", "unknown")
            _min_tf = "5.1.0"
            try:
                from packaging.version import Version
                if tf_version != "unknown" and Version(tf_version) < Version(_min_tf):
                    errors.append(
                        CheckResult(
                            "transformers_version",
                            False,
                            f"transformers>={_min_tf} required for GlmOcrProcessor, "
                            f"but {tf_version} is installed. "
                            f"Run: pip install --no-deps 'transformers>={_min_tf}'",
                        )
                    )
                else:
                    infos.append(
                        CheckResult("transformers_version", True, f"transformers {tf_version} >= {_min_tf}")
                    )
            except Exception:
                warnings.append(
                    CheckResult(
                        "transformers_version",
                        False,
                        f"Could not verify transformers version ({tf_version}); "
                        f">={_min_tf} required for GlmOcrProcessor",
                    )
                )
        except Exception:
            errors.append(
                CheckResult(
                    "transformers",
                    False,
                    "transformers not installed — required for GlmOcrProcessor",
                )
            )

        # Check huggingface-hub version (transformers >= 5.1 requires >= 1.0)
        try:
            import huggingface_hub as _hf  # type: ignore

            hf_version = getattr(_hf, "__version__", "unknown")
            _min_hf = "1.0"
            try:
                from packaging.version import Version
                if hf_version != "unknown" and Version(hf_version) < Version(_min_hf):
                    errors.append(
                        CheckResult(
                            "huggingface_hub_version",
                            False,
                            f"huggingface-hub>={_min_hf} required (transformers 5.x dependency), "
                            f"but {hf_version} is installed. "
                            f"Run: pip install 'huggingface-hub>={_min_hf}'",
                        )
                    )
                else:
                    infos.append(
                        CheckResult("huggingface_hub_version", True, f"huggingface-hub {hf_version} >= {_min_hf}")
                    )
            except Exception:
                warnings.append(
                    CheckResult(
                        "huggingface_hub_version",
                        False,
                        f"Could not verify huggingface-hub version ({hf_version}); "
                        f">={_min_hf} required for transformers 5.x",
                    )
                )
        except Exception:
            errors.append(
                CheckResult(
                    "huggingface_hub",
                    False,
                    "huggingface-hub not installed — required by transformers 5.x",
                )
            )

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
    else:
        warnings.append(
            CheckResult(
                "platform",
                False,
                "GLM-OCR MLX backend is designed for macOS Apple Silicon; "
                "current platform is not macOS.",
            )
        )

    # --- Model weights ---
    model_dir = env.get("GLOSSAPI_GLMOCR_MODEL_DIR", "").strip()
    model_id = env.get("GLOSSAPI_GLMOCR_MLX_MODEL", "").strip()
    # Fallback: check GLOSSAPI_WEIGHTS_ROOT/glm-ocr-mlx/
    if not model_dir:
        from glossapi.ocr.utils.weights import resolve_weights_dir
        resolved = resolve_weights_dir("glm-ocr-mlx", require_config_json=False)
        if resolved is not None:
            model_dir = str(resolved)
    if model_dir:
        md_path = Path(model_dir)
        if not md_path.exists():
            warnings.append(
                CheckResult(
                    "model_dir",
                    False,
                    f"Model dir does not exist: {md_path}",
                )
            )
        else:
            has_config = (md_path / "config.json").exists()
            has_weights = any(md_path.glob("*.safetensors"))
            if not has_weights or not has_config:
                warnings.append(
                    CheckResult(
                        "model_contents",
                        False,
                        f"Model dir {md_path} is missing weights and/or config.json",
                    )
                )
            else:
                infos.append(
                    CheckResult(
                        "model_dir",
                        True,
                        f"Local model dir: {md_path}",
                    )
                )
    elif model_id:
        infos.append(
            CheckResult(
                "model",
                True,
                f"Using HuggingFace model identifier: {model_id}",
            )
        )
    else:
        infos.append(
            CheckResult(
                "model",
                True,
                "Using default HuggingFace model: mlx-community/GLM-OCR-4bit",
            )
        )

    # --- Python binary ---
    python_bin = Path(env.get("GLOSSAPI_GLMOCR_PYTHON") or sys.executable)
    if not python_bin.exists():
        errors.append(
            CheckResult("python", False, f"Python binary not found: {python_bin}")
        )

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_glmocr_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
