"""Tests for the DeepSeek-OCR V1 MLX/MPS runner path.

Covers:
- Multi-root PDF path resolution helpers
- In-process MLX execution (monkeypatched mlx_cli)
- MLX CLI subprocess path (monkeypatched)
- GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI env-var routing
- GLOSSAPI_DEEPSEEK_OCR_DEVICE=mps device selection
- resolve_weights_dir fallback for model_dir in runner
- Stub fallback when MLX fails
- Corpus.ocr() dispatches to V1 runner on MPS
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# run_for_files is imported at module level so every test can call it directly.
# Monkeypatching runner._run_inproc / runner._run_cli_mlx via monkeypatch.setattr
# still works because run_for_files resolves those symbols through the runner
# module's own globals at call-time, not at import-time.
from glossapi.ocr.deepseek_ocr.runner import run_for_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_stub_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n%stub\n")


def _mk_fake_corpus(tmp_path: Path):
    """Return a minimal FakeCorpus-like object with input/output dirs."""

    class FakeCorpus:
        def __init__(self) -> None:
            self.input_dir = str(tmp_path / "in")
            self.output_dir = str(tmp_path / "out")

    Path(FakeCorpus().input_dir).mkdir(parents=True, exist_ok=True)
    Path(FakeCorpus().output_dir).mkdir(parents=True, exist_ok=True)
    return FakeCorpus()


# ---------------------------------------------------------------------------
# 1. Package import
# ---------------------------------------------------------------------------


def test_package_importable_without_heavy_deps():
    """Importing the deepseek_ocr package must not require mlx-vlm or vLLM."""
    import glossapi.ocr.deepseek_ocr as pkg

    assert hasattr(pkg, "run_for_files"), "run_for_files must be exported"
    assert hasattr(pkg, "preflight"), "preflight module must be exported"


def test_runner_has_mlx_helpers():
    """runner.py must expose the multi-root path resolution helpers."""
    from glossapi.ocr.deepseek_ocr import runner

    assert callable(runner._candidate_input_roots)
    assert callable(runner._resolve_pdf_paths)
    assert callable(runner._pick_cli_input_root)


# ---------------------------------------------------------------------------
# 2. Multi-root path resolution helpers
# ---------------------------------------------------------------------------


def test_candidate_input_roots_returns_three_candidates(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import _candidate_input_roots

    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    roots = _candidate_input_roots(input_root, output_root)
    assert roots[0] == input_root
    assert roots[1] == output_root / "downloads"
    assert roots[2] == input_root / "downloads"
    assert len(roots) == 3


def test_resolve_pdf_paths_finds_file_in_downloads(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import _candidate_input_roots, _resolve_pdf_paths

    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    dl_dir = output_root / "downloads"
    dl_dir.mkdir(parents=True)
    pdf = dl_dir / "sample.pdf"
    _write_stub_pdf(pdf)

    roots = _candidate_input_roots(input_root, output_root)
    resolved, missing = _resolve_pdf_paths(["sample.pdf"], roots)
    assert len(resolved) == 1
    assert resolved[0].exists()
    assert missing == [], "file found in downloads should not be in missing"


def test_resolve_pdf_paths_reports_missing(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import _candidate_input_roots, _resolve_pdf_paths

    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    roots = _candidate_input_roots(input_root, output_root)
    resolved, missing = _resolve_pdf_paths(["nonexistent.pdf"], roots)
    assert "nonexistent.pdf" in missing
    assert len(resolved) == 1  # fallback path still returned


def test_resolve_pdf_paths_handles_absolute_path(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import _resolve_pdf_paths

    pdf = tmp_path / "abs.pdf"
    _write_stub_pdf(pdf)
    resolved, missing = _resolve_pdf_paths([str(pdf)], [tmp_path])
    assert resolved[0] == pdf
    assert missing == []


def test_pick_cli_input_root_single_parent(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import _pick_cli_input_root

    dl = tmp_path / "downloads"
    dl.mkdir()
    pdf1 = dl / "a.pdf"
    pdf2 = dl / "b.pdf"
    _write_stub_pdf(pdf1)
    _write_stub_pdf(pdf2)
    root = _pick_cli_input_root(["a.pdf", "b.pdf"], [pdf1, pdf2], [tmp_path, dl])
    assert root == dl


# ---------------------------------------------------------------------------
# 3. Device detection — MPS path activated on macOS
# ---------------------------------------------------------------------------


def test_device_mps_when_env_set(tmp_path, monkeypatch):
    """GLOSSAPI_DEEPSEEK_OCR_DEVICE=mps must select the MPS inference path."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr.runner import run_for_files

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    # Stub allowed — should return without error even on non-macOS CI
    results = run_for_files(fake, ["doc.pdf"], allow_stub=True, allow_inproc=False, allow_mlx_cli=False)
    assert "doc" in results


# ---------------------------------------------------------------------------
# 4. GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI env var routing
# ---------------------------------------------------------------------------


def test_allow_mlx_cli_env_enables_cli(tmp_path, monkeypatch):
    """When GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI=1, the MLX CLI path is attempted."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI", "1")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    cli_called: List[bool] = []

    real_run_cli_mlx = runner._run_cli_mlx

    def fake_run_cli_mlx(input_dir, output_dir, **kwargs):
        cli_called.append(True)
        # Simulate successful CLI run by writing output files
        stem = "doc"
        md_dir = Path(output_dir) / "markdown"
        metrics_dir = Path(output_dir) / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / f"{stem}.md").write_text("mlx cli output\n", encoding="utf-8")
        (metrics_dir / f"{stem}.metrics.json").write_text(
            json.dumps({"page_count": 1}), encoding="utf-8"
        )

    monkeypatch.setattr(runner, "_run_cli_mlx", fake_run_cli_mlx)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: False)  # disable in-process

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    results = run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=True,
        allow_inproc=False,
        allow_mlx_cli=False,  # kwarg says no — but env var should override to yes
    )
    assert cli_called, "MLX CLI should have been called via env var override"
    assert "doc" in results


def test_allow_mlx_cli_env_disables_cli(tmp_path, monkeypatch):
    """When GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI=0, MLX CLI is skipped even if kwarg allows it."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI", "0")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    cli_called: List[bool] = []

    def fail_cli(*args, **kwargs):
        cli_called.append(True)
        raise AssertionError("CLI should not have been called")

    monkeypatch.setattr(runner, "_run_cli_mlx", fail_cli)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: False)

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    results = run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=True,
        allow_inproc=False,
        allow_mlx_cli=True,  # kwarg says yes — but env var overrides to no
    )
    assert not cli_called, "MLX CLI should have been skipped via env var override"
    assert "doc" in results  # stub ran instead


# ---------------------------------------------------------------------------
# 5. In-process MLX execution (macOS fast path)
# ---------------------------------------------------------------------------


def test_inproc_mlx_called_on_macos(tmp_path, monkeypatch):
    """On macOS when mlx_vlm is available, _run_inproc should be invoked."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "0")

    from glossapi.ocr.deepseek_ocr import runner

    inproc_called: List[bool] = []

    def fake_run_inproc(resolved_paths, file_list, out_root, md_dir, metrics_dir, **kwargs):
        inproc_called.append(True)
        stem = "doc"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / f"{stem}.md").write_text("inproc output\n", encoding="utf-8")
        (metrics_dir / f"{stem}.metrics.json").write_text(
            json.dumps({"page_count": 2}), encoding="utf-8"
        )
        return {"doc": {"page_count": 2}}

    monkeypatch.setattr(runner, "_run_inproc", fake_run_inproc)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: True)
    # Force macOS detection regardless of CI platform
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    results = run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=False,
        allow_inproc=True,
        allow_mlx_cli=False,
    )
    assert inproc_called, "_run_inproc must be called on macOS with mlx available"
    assert results["doc"]["page_count"] == 2


def test_inproc_mlx_falls_back_to_stub_on_failure(tmp_path, monkeypatch):
    """When in-process MLX raises, runner falls back to stub if allowed."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    def boom(*args, **kwargs):
        raise RuntimeError("MLX exploded")

    monkeypatch.setattr(runner, "_run_inproc", boom)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: True)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    results = run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=True,
        allow_inproc=True,
        allow_mlx_cli=False,
    )
    assert "doc" in results  # stub took over


def test_inproc_mlx_raises_when_stub_disabled(tmp_path, monkeypatch):
    """When in-process MLX fails and stub is disabled, runner must propagate the error."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "0")

    from glossapi.ocr.deepseek_ocr import runner

    def boom(*args, **kwargs):
        raise RuntimeError("MLX exploded")

    monkeypatch.setattr(runner, "_run_inproc", boom)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: True)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    with pytest.raises(RuntimeError, match="MLX exploded"):
        run_for_files(
            fake,
            ["doc.pdf"],
            allow_stub=False,
            allow_inproc=True,
            allow_mlx_cli=False,
        )


# ---------------------------------------------------------------------------
# 6. resolve_weights_dir fallback in runner.py
# ---------------------------------------------------------------------------


def test_model_dir_resolved_via_weights_root(tmp_path, monkeypatch):
    """runner.py must fall back to resolve_weights_dir('deepseek-ocr-1-mlx')
    when neither GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR nor the model_dir kwarg
    are supplied."""
    # Point GLOSSAPI_WEIGHTS_ROOT at tmp_path so resolve_weights_dir can find the dir
    weights_subdir = tmp_path / "deepseek-ocr-1-mlx"
    weights_subdir.mkdir()
    (weights_subdir / "config.json").write_text("{}", encoding="utf-8")
    (weights_subdir / "model.safetensors").write_bytes(b"stub")

    monkeypatch.setenv("GLOSSAPI_WEIGHTS_ROOT", str(tmp_path))
    monkeypatch.delenv("GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR", raising=False)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    captured_model_dir: List[Optional[Path]] = []

    def fake_run_inproc(resolved_paths, file_list, out_root, md_dir, metrics_dir, *, model_dir, **kwargs):
        captured_model_dir.append(model_dir)
        # Need to raise so it falls through to stub
        raise RuntimeError("capturing model_dir only")

    monkeypatch.setattr(runner, "_run_inproc", fake_run_inproc)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: True)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    # Falls back to stub because _run_inproc raises
    run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=True,
        allow_inproc=True,
        allow_mlx_cli=False,
        model_dir=None,  # explicitly None — must pick up from weights root
    )

    assert captured_model_dir, "_run_inproc was never called"
    assert captured_model_dir[0] == weights_subdir, (
        f"Expected model_dir={weights_subdir!r}, got {captured_model_dir[0]!r}"
    )


# ---------------------------------------------------------------------------
# 7. CLI subprocess path uses _pick_cli_input_root
# ---------------------------------------------------------------------------


def test_cli_input_root_best_match_used(tmp_path, monkeypatch):
    """The MLX CLI subprocess must receive the directory that contains the PDFs."""
    dl = tmp_path / "out" / "downloads"
    dl.mkdir(parents=True)
    pdf = dl / "sample.pdf"
    _write_stub_pdf(pdf)

    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI", "1")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    cli_input_dirs: List[Path] = []

    def fake_run_cli_mlx(input_dir, output_dir, **kwargs):
        cli_input_dirs.append(Path(input_dir))
        stem = "sample"
        md = Path(output_dir) / "markdown"
        metrics = Path(output_dir) / "json" / "metrics"
        md.mkdir(parents=True, exist_ok=True)
        metrics.mkdir(parents=True, exist_ok=True)
        (md / f"{stem}.md").write_text("cli\n", encoding="utf-8")
        (metrics / f"{stem}.metrics.json").write_text(
            json.dumps({"page_count": 1}), encoding="utf-8"
        )

    monkeypatch.setattr(runner, "_run_cli_mlx", fake_run_cli_mlx)
    monkeypatch.setattr(runner, "_can_import_mlx", lambda: False)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    results = run_for_files(
        fake,
        ["sample.pdf"],
        allow_stub=True,
        allow_inproc=False,
        allow_mlx_cli=False,  # env var overrides this to True
        output_dir=tmp_path / "out",
    )
    assert cli_input_dirs, "CLI must have been invoked"
    # The CLI should have been pointed at the downloads dir where the PDF lives
    assert cli_input_dirs[0] == dl, (
        f"Expected CLI input dir {dl!r}, got {cli_input_dirs[0]!r}"
    )


# ---------------------------------------------------------------------------
# 8. Stub fallback — no MPS strategy available
# ---------------------------------------------------------------------------


def test_run_for_files_stub_mps_path(tmp_path, monkeypatch):
    """When on MPS path with no MLX available, stub output is emitted."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "1")

    from glossapi.ocr.deepseek_ocr import runner

    monkeypatch.setattr(runner, "_can_import_mlx", lambda: False)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    results = run_for_files(
        fake,
        ["doc.pdf"],
        allow_stub=True,
        allow_inproc=False,
        allow_mlx_cli=False,
    )
    assert "doc" in results

    md = Path(fake.output_dir) / "markdown" / "doc.md"
    assert md.exists()
    assert "stub" in md.read_text(encoding="utf-8").lower()

    metrics = Path(fake.output_dir) / "json" / "metrics" / "doc.metrics.json"
    assert metrics.exists()
    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "page_count" in data


def test_run_for_files_no_strategy_raises(tmp_path, monkeypatch):
    """When every strategy is disabled + stub is off, a RuntimeError is raised."""
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_DEVICE", "mps")
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB", "0")

    from glossapi.ocr.deepseek_ocr import runner

    monkeypatch.setattr(runner, "_can_import_mlx", lambda: False)
    monkeypatch.setattr("glossapi.ocr.deepseek_ocr.runner.platform.system", lambda: "Darwin")

    fake = _mk_fake_corpus(tmp_path)
    pdf = Path(fake.input_dir) / "doc.pdf"
    _write_stub_pdf(pdf)

    with pytest.raises(RuntimeError, match="no execution strategy"):
        run_for_files(
            fake,
            ["doc.pdf"],
            allow_stub=False,
            allow_inproc=False,
            allow_mlx_cli=False,
        )


# ---------------------------------------------------------------------------
# 9. Corpus.ocr() dispatches to V1 runner on MPS backend
# ---------------------------------------------------------------------------


def test_corpus_ocr_dispatches_to_deepseek_v1_runner(tmp_path, monkeypatch):
    """Corpus.ocr(backend='deepseek-ocr') must invoke the V1 runner.run_for_files."""
    pytest.importorskip("aiohttp", reason="Corpus import requires aiohttp")
    pytest.importorskip("pandas", reason="test requires pandas")
    import pandas as pd
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    corpus = Corpus(input_dir=root, output_dir=root)

    # Seed metadata parquet flagging one file for OCR
    dl_dir = root / "download_results"
    dl_dir.mkdir(parents=True)
    fname = "page.pdf"
    df = pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    )
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)
    (root / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    from glossapi.ocr.deepseek_ocr import runner

    captured: Dict[str, Any] = {}

    def fake_run(self_ref, files, **kwargs):
        captured["files"] = list(files)
        captured["device"] = kwargs.get("device")
        stem = "page"
        md_dir = root / "markdown"
        m_dir = root / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        m_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / f"{stem}.md").write_text("deep ocr\n", encoding="utf-8")
        (m_dir / f"{stem}.metrics.json").write_text(
            json.dumps({"page_count": 1}), encoding="utf-8"
        )
        return {stem: {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run)

    corpus.ocr(backend="deepseek-ocr", fix_bad=True, math_enhance=False)

    assert captured.get("files") == [fname], "runner must receive the bad file"
    # Parquet updated
    updated = pd.read_parquet(parquet_path).set_index("filename")
    assert bool(updated.loc[fname, "ocr_success"]) is True
    assert bool(updated.loc[fname, "needs_ocr"]) is False


# ---------------------------------------------------------------------------
# 10. run_for_files with empty file list
# ---------------------------------------------------------------------------


def test_run_for_files_empty_returns_empty(tmp_path):
    from glossapi.ocr.deepseek_ocr.runner import run_for_files

    class Fake:
        input_dir = str(tmp_path)
        output_dir = str(tmp_path)

    assert run_for_files(Fake(), []) == {}
