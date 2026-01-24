"""MinerU (Magic-PDF) OCR runner with optional CLI dispatch."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
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
) -> None:
    cmd: List[str] = [
        magic_pdf_bin,
        "-p",
        str(pdf_path),
        "-o",
        str(tmp_out),
        "-m",
        mode,
    ]
    LOGGER.info("Running MinerU CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # nosec: controlled arguments


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
    allow_stub: bool = True,
    allow_cli: bool = False,
    magic_pdf_bin: Optional[str] = None,
    mode: Optional[str] = None,
    content_debug: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """Run MinerU OCR for the provided files.

    Returns a mapping of stem -> minimal metadata (page_count).
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    tmp_root = out_root / "mineru_tmp"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ
    env_allow_stub = env.get("GLOSSAPI_MINERU_ALLOW_STUB", "1") == "1"
    env_allow_cli = env.get("GLOSSAPI_MINERU_ALLOW_CLI", "0") == "1"
    env_mode = env.get("GLOSSAPI_MINERU_MODE")
    env_bin = env.get("GLOSSAPI_MINERU_COMMAND")

    use_cli = allow_cli or env_allow_cli
    use_stub = allow_stub and env_allow_stub
    mode = (mode or env_mode or "auto").strip()
    magic_pdf = _resolve_magic_pdf(magic_pdf_bin or env_bin)

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
                _run_cli(pdf_path, tmp_out, magic_pdf_bin=magic_pdf, mode=mode)
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
