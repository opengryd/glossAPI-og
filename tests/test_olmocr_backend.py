"""Tests for the OlmOCR-2 OCR backend (stub / dispatch)."""

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


def test_olmocr_package_importable():
    """The OlmOCR sub-package must be importable without heavy deps."""
    import glossapi.ocr.olmocr as olmocr_pkg

    assert hasattr(olmocr_pkg, "run_for_files")
    assert hasattr(olmocr_pkg, "preflight")


def test_olmocr_lazy_alias():
    """The OCR __init__.py lazy alias must resolve."""
    import glossapi.ocr as ocr

    assert hasattr(ocr, "olmocr")
    assert ocr.olmocr_runner is ocr.olmocr.runner


# ---- Preflight tests ----


def test_preflight_stub_warnings(monkeypatch):
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "0",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "1",
    }
    report = check_olmocr_env(env, check_gpu=False)
    names = [w.name for w in report.warnings]
    assert "enable_ocr" in names
    assert "enable_stub" in names


def test_preflight_model_dir_missing(tmp_path, monkeypatch):
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_OLMOCR_MODEL_DIR": str(tmp_path / "nonexistent"),
    }
    report = check_olmocr_env(env, check_gpu=False)
    error_names = [e.name for e in report.errors]
    assert "model_dir" in error_names


# ---- Runner stub tests ----


def test_run_for_files_stub_output(tmp_path):
    """With stubs allowed (default), run_for_files should produce markdown + metrics."""
    from glossapi.ocr.olmocr.runner import run_for_files

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
    assert "OlmOCR (stub)" in content

    metrics_file = output_dir / "json" / "metrics" / "sample.metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "page_count" in metrics


def test_run_for_files_empty_list(tmp_path):
    """Passing no files should return an empty dict."""
    from glossapi.ocr.olmocr.runner import run_for_files

    class FakeCorpus:
        input_dir = str(tmp_path)
        output_dir = str(tmp_path)

    assert run_for_files(FakeCorpus(), []) == {}


def test_run_for_files_stub_disabled_no_cli_raises(tmp_path, monkeypatch):
    """When both stub and CLI are disabled, runner should raise."""
    from glossapi.ocr.olmocr.runner import run_for_files

    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_STUB", "0")
    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_OCR", "0")

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
            enable_mlx_ocr=False,
        )


def test_run_for_files_content_debug(tmp_path):
    """content_debug should produce a debug comment in the stub markdown."""
    from glossapi.ocr.olmocr.runner import run_for_files

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


def test_olmocr_backend_routes_to_runner(tmp_path, monkeypatch):
    """Corpus.ocr(backend='olmocr') should dispatch through the olmocr runner."""
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
    from glossapi.ocr.olmocr import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    corpus.ocr(backend="olmocr", fix_bad=True, math_enhance=False, mode="ocr_bad")

    assert calls.get("files") == [fname]


def test_olmocr_backend_invalid_raises():
    """An invalid backend name should raise ValueError."""
    from glossapi import Corpus

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        corpus = Corpus(input_dir=td, output_dir=td)
        with pytest.raises(ValueError, match="backend must be"):
            corpus.ocr(backend="nonexistent")


# ---- MLX / MPS tests ----


def test_can_import_mlx():
    """_can_import_mlx returns a bool without raising."""
    from glossapi.ocr.olmocr.runner import _can_import_mlx

    result = _can_import_mlx()
    assert isinstance(result, bool)


def test_mlx_cli_module_importable():
    """The mlx_cli module must be importable (no heavy deps at import time)."""
    from glossapi.ocr.olmocr import mlx_cli

    assert hasattr(mlx_cli, "load_model_and_processor")
    assert hasattr(mlx_cli, "resolve_model_dir")
    assert hasattr(mlx_cli, "process_pdf")
    assert hasattr(mlx_cli, "generate_page")
    assert hasattr(mlx_cli, "render_page")
    assert hasattr(mlx_cli, "main")
    assert hasattr(mlx_cli, "DEFAULT_MODEL_ID")
    assert hasattr(mlx_cli, "DEFAULT_PROMPT")


def test_mlx_cli_default_constants():
    """MLX CLI constants should have sensible defaults."""
    from glossapi.ocr.olmocr.mlx_cli import (
        DEFAULT_DPI,
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_ID,
        DEFAULT_PROMPT,
    )

    assert DEFAULT_DPI == 200
    assert DEFAULT_MAX_TOKENS == 4096
    assert "mlx-community" in DEFAULT_MODEL_ID
    assert "olmOCR" in DEFAULT_MODEL_ID
    assert "markdown" in DEFAULT_PROMPT.lower()


def test_run_for_files_inproc_skipped_on_non_darwin(tmp_path, monkeypatch):
    """In-process MLX should be skipped on non-macOS platforms if forced."""
    from glossapi.ocr.olmocr.runner import run_for_files

    # Even if _can_import_mlx returns True, non-Darwin should skip in-process
    monkeypatch.setattr("glossapi.ocr.olmocr.runner.platform.system", lambda: "Linux")

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    # Should fall through to stub (default enabled)
    results = run_for_files(
        FakeCorpus(), ["doc.pdf"], enable_inproc=True, enable_mlx_ocr=False
    )
    assert "doc" in results

    # Verify stub output (not MLX)
    md = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert "stub" in md.lower()


def test_run_for_files_all_strategies_disabled_raises(tmp_path, monkeypatch):
    """When all strategies are disabled, runner should raise RuntimeError."""
    from glossapi.ocr.olmocr.runner import run_for_files

    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_STUB", "0")
    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_OCR", "0")
    monkeypatch.setattr("glossapi.ocr.olmocr.runner.platform.system", lambda: "Linux")

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
            enable_mlx_ocr=False,
        )


def test_preflight_mlx_checks_on_macos(monkeypatch):
    """On macOS, preflight should include MLX-related info/warning checks."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Darwin")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
    }
    report = check_olmocr_env(env, check_gpu=False)

    # Should have either mlx_vlm info or warning
    all_names = (
        [c.name for c in report.infos]
        + [c.name for c in report.warnings]
        + [c.name for c in report.errors]
    )
    assert "mlx_vlm" in all_names


def test_preflight_mlx_model_dir_missing(tmp_path, monkeypatch):
    """Preflight should warn about missing MLX model directory."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Darwin")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_OLMOCR_MLX_MODEL_DIR": str(tmp_path / "nonexistent_mlx"),
    }
    report = check_olmocr_env(env, check_gpu=False)
    warn_names = [w.name for w in report.warnings]
    assert "mlx_model_dir" in warn_names


def test_preflight_mlx_model_dir_valid(tmp_path, monkeypatch):
    """Preflight should accept a valid MLX model directory."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Darwin")

    model_dir = tmp_path / "mlx_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_OLMOCR_MLX_MODEL_DIR": str(model_dir),
    }
    report = check_olmocr_env(env, check_gpu=False)
    info_names = [i.name for i in report.infos]
    assert "mlx_model_dir" in info_names


def test_preflight_mlx_model_id_env(monkeypatch):
    """Preflight should report custom MLX model ID."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Darwin")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_OLMOCR_MLX_MODEL": "custom-repo/olmocr-mlx-4bit",
    }
    report = check_olmocr_env(env, check_gpu=False)
    info_names = [i.name for i in report.infos]
    assert "mlx_model" in info_names
    mlx_info = [i for i in report.infos if i.name == "mlx_model"]
    assert "custom-repo/olmocr-mlx-4bit" in mlx_info[0].message


# ---- CUDA / vLLM tests ----


def test_can_import_vllm():
    """_can_import_vllm returns a bool without raising."""
    from glossapi.ocr.olmocr.runner import _can_import_vllm

    result = _can_import_vllm()
    assert isinstance(result, bool)


def test_cuda_available():
    """_cuda_available returns a bool without raising."""
    from glossapi.ocr.olmocr.runner import _cuda_available

    result = _cuda_available()
    assert isinstance(result, bool)


def test_vllm_cli_module_importable():
    """The vllm_cli module must be importable (no heavy deps at import time)."""
    from glossapi.ocr.olmocr import vllm_cli

    assert hasattr(vllm_cli, "load_model")
    assert hasattr(vllm_cli, "resolve_model_dir")
    assert hasattr(vllm_cli, "process_pdf")
    assert hasattr(vllm_cli, "generate_page")
    assert hasattr(vllm_cli, "render_page")
    assert hasattr(vllm_cli, "main")
    assert hasattr(vllm_cli, "DEFAULT_MODEL_ID")
    assert hasattr(vllm_cli, "DEFAULT_PROMPT")
    assert hasattr(vllm_cli, "DEFAULT_GPU_MEMORY_UTILIZATION")
    assert hasattr(vllm_cli, "DEFAULT_TENSOR_PARALLEL_SIZE")
    assert hasattr(vllm_cli, "DEFAULT_MAX_MODEL_LEN")


def test_vllm_cli_default_constants():
    """vLLM CLI constants should have sensible defaults."""
    from glossapi.ocr.olmocr.vllm_cli import (
        DEFAULT_DPI,
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_ID,
        DEFAULT_PROMPT,
        DEFAULT_GPU_MEMORY_UTILIZATION,
        DEFAULT_TENSOR_PARALLEL_SIZE,
        DEFAULT_MAX_MODEL_LEN,
    )

    assert DEFAULT_DPI == 150
    assert DEFAULT_MAX_TOKENS == 2048
    assert "olmOCR" in DEFAULT_MODEL_ID
    assert "markdown" in DEFAULT_PROMPT.lower()
    assert 0 < DEFAULT_GPU_MEMORY_UTILIZATION <= 1.0
    assert DEFAULT_TENSOR_PARALLEL_SIZE >= 1
    assert DEFAULT_MAX_MODEL_LEN > 0


def test_run_for_files_inproc_vllm_skipped_on_darwin(tmp_path, monkeypatch):
    """In-process vLLM should be skipped on macOS platforms."""
    from glossapi.ocr.olmocr.runner import run_for_files

    monkeypatch.setattr("glossapi.ocr.olmocr.runner.platform.system", lambda: "Darwin")

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    # Should fall through to stub since Darwin skips vLLM strategies
    results = run_for_files(
        FakeCorpus(),
        ["doc.pdf"],
        enable_inproc=False,
        enable_mlx_ocr=False,
        enable_inproc_vllm=True,
        enable_vllm_cli=True,
    )
    assert "doc" in results
    md = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert "stub" in md.lower()


def test_run_for_files_vllm_strategies_on_linux(tmp_path, monkeypatch):
    """On Linux without vLLM, in-process vLLM should be skipped gracefully."""
    from glossapi.ocr.olmocr.runner import run_for_files

    monkeypatch.setattr("glossapi.ocr.olmocr.runner.platform.system", lambda: "Linux")
    monkeypatch.setattr("glossapi.ocr.olmocr.runner._can_import_vllm", lambda: False)
    monkeypatch.setattr("glossapi.ocr.olmocr.runner._cuda_available", lambda: False)

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    class FakeCorpus:
        def __init__(self):
            self.input_dir = str(input_dir)
            self.output_dir = str(output_dir)

    # Should fall through to stub
    results = run_for_files(
        FakeCorpus(),
        ["doc.pdf"],
        enable_inproc=False,
        enable_mlx_ocr=False,
        enable_inproc_vllm=True,
        enable_vllm_cli=False,
        enable_ocr=False,
    )
    assert "doc" in results
    md = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert "stub" in md.lower()


def test_run_for_files_all_six_strategies_disabled_raises(tmp_path, monkeypatch):
    """When all six strategies are disabled, runner should raise RuntimeError."""
    from glossapi.ocr.olmocr.runner import run_for_files

    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_STUB", "0")
    monkeypatch.setenv("GLOSSAPI_OLMOCR_ENABLE_OCR", "0")
    monkeypatch.setattr("glossapi.ocr.olmocr.runner.platform.system", lambda: "Linux")

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
            enable_mlx_ocr=False,
            enable_inproc_vllm=False,
            enable_vllm_cli=False,
        )


def test_preflight_vllm_check_on_linux(monkeypatch):
    """On Linux, preflight should include vLLM-related info/warning checks."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Linux")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
    }
    report = check_olmocr_env(env, check_gpu=False)
    all_names = (
        [c.name for c in report.infos]
        + [c.name for c in report.warnings]
        + [c.name for c in report.errors]
    )
    assert "vllm" in all_names


def test_preflight_cuda_model_dir_valid(tmp_path, monkeypatch):
    """Preflight should accept a valid CUDA model directory."""
    from glossapi.ocr.olmocr.preflight import check_olmocr_env

    monkeypatch.setattr("glossapi.ocr.olmocr.preflight.platform.system", lambda: "Linux")

    model_dir = tmp_path / "cuda_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00")

    env = {
        "GLOSSAPI_OLMOCR_ENABLE_OCR": "1",
        "GLOSSAPI_OLMOCR_ENABLE_STUB": "0",
        "GLOSSAPI_OLMOCR_MODEL_DIR": str(model_dir),
    }
    report = check_olmocr_env(env, check_gpu=False)
    info_names = [i.name for i in report.infos]
    assert "cuda_model_dir" in info_names
