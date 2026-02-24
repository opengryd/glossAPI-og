"""MinerU (Magic-PDF) OCR runner with optional CLI dispatch."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import prepare_env_with_config

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)
_MAGIC_PDF_BACKEND_SUPPORT: Optional[bool] = None


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        _doc = _pypdfium2.PdfDocument(str(pdf_path))
        try:
            return len(_doc)
        finally:
            _doc.close()
    except Exception:
        return 0


def _resolve_magic_pdf(cmd: Optional[str]) -> Optional[str]:
    if cmd:
        return cmd
    return shutil.which("magic-pdf")


def _find_markdown(tmp_out: Path, stem: str) -> Optional[Path]:
    if not tmp_out.exists():
        return None
    candidates = sorted(tmp_out.rglob(f"{stem}.md"))
    if candidates:
        return candidates[0]
    candidates = sorted(tmp_out.rglob("*.md"))
    if candidates:
        return candidates[0]
    return None


def _run_cli(
    pdf_path: Path,
    tmp_out: Path,
    *,
    magic_pdf_bin: str,
    mode: str,
    backend: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    supports_backend = _supports_backend_flag(magic_pdf_bin)
    cmd: List[str] = [
        magic_pdf_bin,
        "-p",
        str(pdf_path),
        "-o",
        str(tmp_out),
        "-m",
        mode,
    ]
    if backend and supports_backend:
        cmd.extend(["-b", backend])
    elif backend and not supports_backend:
        LOGGER.info("MinerU CLI does not support -b/--backend; ignoring backend=%s", backend)
    LOGGER.info("Running MinerU CLI: %s", " ".join(cmd))
    # Suppress magic-pdf's loguru INFO/WARNING noise; only propagate ERROR+.
    quiet_env = dict(env) if env else dict(os.environ)
    quiet_env.setdefault("LOGURU_LEVEL", "ERROR")
    subprocess.run(cmd, check=True, env=quiet_env)  # nosec: controlled arguments


def _supports_backend_flag(magic_pdf_bin: str) -> bool:
    global _MAGIC_PDF_BACKEND_SUPPORT
    if _MAGIC_PDF_BACKEND_SUPPORT is not None:
        return _MAGIC_PDF_BACKEND_SUPPORT
    try:
        result = subprocess.run(
            [magic_pdf_bin, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        _MAGIC_PDF_BACKEND_SUPPORT = "-b," in output or "--backend" in output
    except Exception:
        _MAGIC_PDF_BACKEND_SUPPORT = False
    return _MAGIC_PDF_BACKEND_SUPPORT


def _normalize_device_mode(device: Optional[str]) -> Optional[str]:
    if not device:
        return None
    value = str(device).strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value in {"mps", "metal"}:
        return "mps"
    if value in {"cpu"}:
        return "cpu"
    if value in {"auto"}:
        return None
    return value


def _mps_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _resolve_backend(device_mode: Optional[str], backend: Optional[str]) -> Optional[str]:
    if backend:
        value = str(backend).strip()
        if value.lower() == "auto":
            return None
        return value
    if device_mode in {"cuda", "mps"}:
        if device_mode == "mps":
            if platform.system() == "Darwin" and _mps_available():
                return "hybrid-auto-engine"
            return "pipeline"
        return "hybrid-auto-engine"
    return "pipeline"


def _prepare_mineru_env(
    base_env: Dict[str, str],
    tmp_root: Path,
    device_mode: Optional[str],
) -> Dict[str, str]:
    """Resolve and set ``MINERU_TOOLS_CONFIG_JSON`` in the subprocess env.

    Uses :func:`config.prepare_env_with_config` to auto-discover the repo's
    ``magic-pdf.json``, resolve relative model paths, and optionally override
    the device mode — all in one step.
    """
    env_out, resolved = prepare_env_with_config(
        base_env,
        tmp_root,
        device_mode=device_mode,
    )
    if resolved is None:
        LOGGER.warning(
            "No MinerU config found (set MINERU_TOOLS_CONFIG_JSON or "
            "place magic-pdf.json in model_weights/mineru/)"
        )
    return env_out


def _write_stub(pdf_path: Path, md_out: Path, metrics_out: Path, content_debug: bool) -> Dict[str, Any]:
    page_count = _page_count(pdf_path)
    md_lines = [
        f"# MinerU OCR (stub) — {pdf_path.name}",
        "",
        f"Pages: {page_count if page_count else 'unknown'}",
    ]
    if content_debug:
        md_lines.append("")
        md_lines.append("<!-- content_debug: stub output -->")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metrics = {"page_count": page_count}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    output_dir: Optional[Path] = None,
    max_pages: Optional[int] = None,
    enable_stub: bool = True,
    enable_ocr: bool = False,
    magic_pdf_bin: Optional[str] = None,
    mode: Optional[str] = None,
    backend: Optional[str] = None,
    content_debug: bool = False,
    device: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run MinerU OCR for the provided files.

    Returns a mapping of stem -> minimal metadata (page_count).
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = (Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))).resolve()
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    tmp_root = out_root / "mineru_tmp"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ
    env_enable_stub = env.get("GLOSSAPI_MINERU_ENABLE_STUB", "1") == "1"
    env_enable_ocr = env.get("GLOSSAPI_MINERU_ENABLE_OCR", "0") == "1"
    env_mode = env.get("GLOSSAPI_MINERU_MODE")
    env_bin = env.get("GLOSSAPI_MINERU_COMMAND")
    env_backend = env.get("GLOSSAPI_MINERU_BACKEND")
    env_device = env.get("GLOSSAPI_MINERU_DEVICE_MODE") or env.get("GLOSSAPI_MINERU_DEVICE")

    use_cli = enable_ocr or env_enable_ocr
    use_stub = enable_stub and env_enable_stub
    mode = (mode or env_mode or "auto").strip()
    magic_pdf = _resolve_magic_pdf(magic_pdf_bin or env_bin)
    device_mode = _normalize_device_mode(device or env_device)
    backend_choice = _resolve_backend(device_mode, backend or env_backend)

    if not use_cli:
        LOGGER.warning(
            "MinerU CLI is disabled (GLOSSAPI_MINERU_ENABLE_OCR=0). "
            "Stub output will be produced. Set GLOSSAPI_MINERU_ENABLE_OCR=1 to run real OCR."
        )
    elif not magic_pdf:
        LOGGER.warning(
            "GLOSSAPI_MINERU_ENABLE_OCR is set but 'magic-pdf' binary was not found. "
            "Install MinerU (pip install mineru) or set GLOSSAPI_MINERU_COMMAND to the "
            "full path of the magic-pdf executable. Falling back to stub output."
        )

    cli_env = _prepare_mineru_env(dict(env), tmp_root, device_mode) if use_cli else env

    results: Dict[str, Any] = {}

    for name in file_list:
        pdf_path = (input_root / name).resolve()
        if not pdf_path.exists():
            dl_candidate = out_root / "downloads" / name
            if dl_candidate.exists():
                pdf_path = dl_candidate.resolve()
            else:
                input_dl = input_root / "downloads" / name
                if input_dl.exists():
                    pdf_path = input_dl.resolve()
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        tmp_out = tmp_root / stem

        if use_cli and magic_pdf:
            try:
                if tmp_out.exists():
                    shutil.rmtree(tmp_out)
                tmp_out.mkdir(parents=True, exist_ok=True)
                _run_cli(
                    pdf_path,
                    tmp_out,
                    magic_pdf_bin=magic_pdf,
                    mode=mode,
                    backend=backend_choice,
                    env=cli_env,
                )
                md_src = _find_markdown(tmp_out, stem)
                if md_src and md_src.exists():
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(md_src, md_path)
                else:
                    raise FileNotFoundError("MinerU output markdown not found")
            except Exception as exc:
                if not use_stub:
                    raise
                LOGGER.warning("MinerU CLI failed (%s); falling back to stub output", exc)
                results[stem] = _write_stub(pdf_path, md_path, metrics_path, content_debug)
                continue
        elif not use_stub:
            raise RuntimeError("MinerU CLI disabled and stub output not allowed")
        else:
            results[stem] = _write_stub(pdf_path, md_path, metrics_path, content_debug)
            continue

        page_count = _page_count(pdf_path)
        metrics = {"page_count": page_count}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        results[stem] = metrics

    return results
