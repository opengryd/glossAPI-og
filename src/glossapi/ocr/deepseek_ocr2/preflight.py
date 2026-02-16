"""Preflight checks for the DeepSeek OCR v2 (MLX/MPS) environment."""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# The CLI script is now shipped inside the package.
_PACKAGE_CLI_SCRIPT = Path(__file__).resolve().parent / "mlx_cli.py"


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


def _has_mlx_weights(model_dir: Path) -> bool:
    patterns = ["*.safetensors", "*.npz", "*.bin", "*.mlx", "*.pt"]
    for pattern in patterns:
        if any(model_dir.glob(pattern)):
            return True
    return False


def check_deepseek_ocr2_env(
    env: Optional[Dict[str, str]] = None,
    *,
    check_mlx: bool = True,
) -> PreflightReport:
    """Validate DeepSeek OCR v2 prerequisites without running the model."""

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    allow_cli = env.get("GLOSSAPI_DEEPSEEK2_ALLOW_CLI", "0") == "1"
    allow_stub = env.get("GLOSSAPI_DEEPSEEK2_ALLOW_STUB", "1") == "1"
    if not allow_cli:
        warnings.append(
            CheckResult(
                "allow_cli",
                False,
                "Set GLOSSAPI_DEEPSEEK2_ALLOW_CLI=1 to force the real MLX CLI.",
            )
        )
    if allow_stub:
        warnings.append(
            CheckResult(
                "allow_stub",
                False,
                "Set GLOSSAPI_DEEPSEEK2_ALLOW_STUB=0 to fail instead of falling back to stub output.",
            )
        )

    if platform.system() != "Darwin":
        errors.append(CheckResult("platform", False, "DeepSeek OCR v2 requires macOS (Metal/MPS)."))

    # CLI script: env var > package-embedded script
    script_env = env.get("GLOSSAPI_DEEPSEEK2_MLX_SCRIPT", "").strip()
    script = Path(script_env) if script_env else _PACKAGE_CLI_SCRIPT
    _ensure_path(script, "mlx_script", errors)

    python_bin = Path(env.get("GLOSSAPI_DEEPSEEK2_TEST_PYTHON") or sys.executable)
    _ensure_path(python_bin, "deepseek2_python", errors)

    # Model dir: env var wins, then GLOSSAPI_WEIGHTS_ROOT, otherwise warn (auto-download at runtime)
    model_dir_env = env.get("GLOSSAPI_DEEPSEEK2_MODEL_DIR", "").strip()
    model_dir: Optional[Path] = None
    if model_dir_env:
        model_dir = _ensure_path(Path(model_dir_env), "model_dir", errors)
    else:
        from glossapi.ocr.utils.weights import resolve_weights_dir
        resolved = resolve_weights_dir("deepseek-ocr-mlx")
        if resolved is not None:
            model_dir = resolved
        else:
            infos.append(
                CheckResult(
                    "model_dir",
                    True,
                    "No model dir found via GLOSSAPI_DEEPSEEK2_MODEL_DIR or GLOSSAPI_WEIGHTS_ROOT; model will be auto-downloaded from HuggingFace at runtime.",
                )
            )
    if model_dir:
        has_config = (model_dir / "config.json").exists()
        has_weights = _has_mlx_weights(model_dir)
        if not has_weights or not has_config:
            errors.append(
                CheckResult(
                    "model_contents",
                    False,
                    f"Model dir {model_dir} is missing weights and/or config.json",
                )
            )

    if check_mlx:
        try:
            import mlx_vlm  # type: ignore

            infos.append(CheckResult("mlx_vlm", True, "mlx-vlm import ok"))
        except Exception as exc:  # pragma: no cover - depends on env
            errors.append(CheckResult("mlx_vlm", False, f"mlx-vlm import failed: {exc}"))

    device = (env.get("GLOSSAPI_DEEPSEEK2_DEVICE") or "mps").strip().lower()
    if device not in {"mps", "cpu"}:
        warnings.append(CheckResult("device", False, f"Unknown device '{device}', expected mps or cpu"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_deepseek_ocr2_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
