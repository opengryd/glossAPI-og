"""Tests for the GLM-OCR backend (stub / dispatch)."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


# ---- Import / lazy-load tests ----


def test_glmocr_package_importable():
    """The GLM-OCR sub-package must be importable without heavy deps."""
    import glossapi.ocr.glm_ocr as glm_ocr_pkg

    assert hasattr(glm_ocr_pkg, "run_for_files")
    assert hasattr(glm_ocr_pkg, "preflight")


def test_glmocr_lazy_alias():
    """The OCR __init__.py lazy alias must resolve."""
    import glossapi.ocr as ocr

    assert hasattr(ocr, "glm_ocr")
    assert ocr.glm_ocr_runner is ocr.glm_ocr.runner


# ---- Preflight tests ----


def test_preflight_stub_warnings(monkeypatch):
    from glossapi.ocr.glm_ocr.preflight import check_glmocr_env

    env = {
        "GLOSSAPI_GLMOCR_ENABLE_OCR": "0",
        "GLOSSAPI_GLMOCR_ENABLE_STUB": "1",
    }
    report = check_glmocr_env(env)
    names = [w.name for w in report.warnings]
    assert "enable_ocr" in names
    assert "enable_stub" in names


def test_preflight_model_dir_missing(tmp_path, monkeypatch):
    from glossapi.ocr.glm_ocr.preflight import check_glmocr_env

    env = {
        "GLOSSAPI_GLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_GLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_GLMOCR_MODEL_DIR": str(tmp_path / "nonexistent"),
    }
    report = check_glmocr_env(env)
    warning_names = [w.name for w in report.warnings]
    assert "model_dir" in warning_names


# ---- Runner stub tests ----


def test_run_for_files_stub_output(tmp_path):
    """With stubs allowed (default), run_for_files should produce markdown + metrics."""
    from glossapi.ocr.glm_ocr.runner import run_for_files

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a minimal fake PDF
    pdf = input_dir / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    results = run_for_files(FakeCorpus(), ["sample.pdf"])

    assert "sample" in results
    assert results["sample"]["page_count"] >= 0

    md_file = output_dir / "markdown" / "sample.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "GLM-OCR (stub)" in content

    metrics_file = output_dir / "json" / "metrics" / "sample.metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "page_count" in metrics


def test_run_for_files_empty_list(tmp_path):
    """Passing no files should return an empty dict."""
    from glossapi.ocr.glm_ocr.runner import run_for_files

    class FakeCorpus:
        input_dir = str(tmp_path)
        output_dir = str(tmp_path)

    assert run_for_files(FakeCorpus(), []) == {}


def test_run_for_files_stub_disabled_no_cli_raises(tmp_path, monkeypatch):
    """When both stub and CLI are disabled, runner should raise."""
    from glossapi.ocr.glm_ocr.runner import run_for_files

    monkeypatch.setenv("GLOSSAPI_GLMOCR_ENABLE_STUB", "0")
    monkeypatch.setenv("GLOSSAPI_GLMOCR_ENABLE_OCR", "0")

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    with pytest.raises(RuntimeError, match="no execution strategy"):
        run_for_files(
            FakeCorpus(),
            ["doc.pdf"],
            enable_stub=False,
            enable_ocr=False,
            enable_inproc=False,
        )


def test_run_for_files_content_debug(tmp_path):
    """content_debug should produce a debug comment in the stub markdown."""
    from glossapi.ocr.glm_ocr.runner import run_for_files

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    run_for_files(FakeCorpus(), ["doc.pdf"], content_debug=True)

    md = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert "content_debug" in md


# ---- Dispatch integration test ----


def test_glmocr_backend_routes_to_runner(tmp_path, monkeypatch):
    """Corpus.ocr(backend='glm-ocr') should dispatch through the glm_ocr runner."""
    corpus = _mk_corpus(tmp_path)

    # Seed metadata with one bad file
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame([
        {"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)

    # Create stub pdf
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    # Capture runner calls
    from glossapi.ocr.glm_ocr import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    corpus.ocr(backend="glm-ocr", fix_bad=True, math_enhance=False, mode="ocr_bad")

    assert calls.get("files") == [fname]


def test_glmocr_backend_invalid_raises():
    """An invalid backend string must raise ValueError."""
    from glossapi import Corpus

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        c = Corpus(input_dir=td, output_dir=td)
        with pytest.raises(ValueError, match="backend must be"):
            c.ocr(backend="invalid_backend")
