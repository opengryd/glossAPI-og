"""Preflight checks for the MinerU (magic-pdf) CLI environment."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
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


def _ensure_path(path: Path, label: str, errors: List[CheckResult]) -> Optional[Path]:
    if not path:
        errors.append(CheckResult(label, False, "Not provided"))
        return None
    if not path.exists():
        errors.append(CheckResult(label, False, f"Missing at {path}"))
        return None
    return path


def _torch_device_available(device_mode: str) -> bool:
    try:
        import torch  # type: ignore

        if device_mode == "mps":
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if device_mode == "cuda":
            return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False
    return True


def _extract_models_dirs(config: dict) -> List[Path]:
    value = config.get("models-dir")
    if isinstance(value, dict):
        return [Path(v) for v in value.values() if v]
    if isinstance(value, str):
        return [Path(value)]
    return []


def check_mineru_env(env: Optional[Dict[str, str]] = None) -> PreflightReport:
    """Validate MinerU CLI prerequisites without running OCR."""

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    allow_cli = env.get("GLOSSAPI_MINERU_ALLOW_CLI", "0") == "1"
    allow_stub = env.get("GLOSSAPI_MINERU_ALLOW_STUB", "1") == "1"
    if not allow_cli:
        warnings.append(
            CheckResult(
                "allow_cli",
                False,
                "Set GLOSSAPI_MINERU_ALLOW_CLI=1 to force the real CLI.",
            )
        )
    if allow_stub:
        warnings.append(
            CheckResult(
                "allow_stub",
                False,
                "Set GLOSSAPI_MINERU_ALLOW_STUB=0 to fail instead of falling back to stub output.",
            )
        )

    magic_pdf = env.get("GLOSSAPI_MINERU_COMMAND") or shutil.which("magic-pdf")
    if not magic_pdf:
        errors.append(CheckResult("magic_pdf", False, "magic-pdf not found on PATH or GLOSSAPI_MINERU_COMMAND"))
    else:
        infos.append(CheckResult("magic_pdf", True, f"Found at {magic_pdf}"))

    config_path = env.get("MINERU_TOOLS_CONFIG_JSON")
    config_file = _ensure_path(Path(config_path) if config_path else Path(""), "config_json", errors)
    config: dict = {}
    if config_file:
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(CheckResult("config_json", False, f"Failed to parse config: {exc}"))

    device_mode = (env.get("GLOSSAPI_MINERU_DEVICE_MODE") or env.get("GLOSSAPI_MINERU_DEVICE") or "").strip().lower()
    if not device_mode and config:
        device_mode = str(config.get("device-mode", "")).strip().lower()
    if device_mode:
        if device_mode not in {"mps", "cuda", "cpu"}:
            warnings.append(CheckResult("device_mode", False, f"Unknown device-mode '{device_mode}'"))
        else:
            if device_mode in {"mps", "cuda"} and not _torch_device_available(device_mode):
                warnings.append(
                    CheckResult(
                        "device_mode",
                        False,
                        f"Torch reports {device_mode} unavailable; MinerU may fall back to CPU",
                    )
                )
            else:
                infos.append(CheckResult("device_mode", True, f"Using device-mode={device_mode}"))

    backend = (env.get("GLOSSAPI_MINERU_BACKEND") or "").strip()
    if backend:
        if backend not in {"pipeline", "hybrid-auto-engine", "vlm", "auto"}:
            warnings.append(CheckResult("backend", False, f"Unknown backend '{backend}'"))
        else:
            infos.append(CheckResult("backend", True, f"Requested backend={backend}"))

    if config:
        model_dirs = _extract_models_dirs(config)
        if model_dirs:
            for idx, model_dir in enumerate(model_dirs, start=1):
                _ensure_path(model_dir, f"models_dir_{idx}", errors)
        else:
            warnings.append(CheckResult("models_dir", False, "No models-dir configured in MinerU config"))

    if platform.system() == "Darwin":
        version = platform.mac_ver()[0]
        if version:
            infos.append(CheckResult("macos", True, f"macOS {version}"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_mineru_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())