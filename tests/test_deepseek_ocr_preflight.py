import sys
from pathlib import Path

from glossapi.ocr.deepseek_ocr.preflight import check_deepseek_ocr_env


def test_preflight_reports_missing_components(tmp_path):
    env = {
        "GLOSSAPI_DEEPSEEK_OCR_DEVICE": "cuda",  # force CUDA path regardless of platform
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR": "0",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB": "1",
        "GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON": str(tmp_path / "missing_python"),
        "GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT": str(tmp_path / "missing_script.py"),
        "GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR": str(tmp_path / "missing_model"),
        "GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH": str(tmp_path / "missing_lib"),
        "PATH": str(tmp_path),  # no cc1plus here
    }
    report = check_deepseek_ocr_env(env, check_flashinfer=False)
    names = {c.name for c in report.errors}
    assert "deepseek_ocr_python" in names
    assert "vllm_script" in names
    assert "model_dir" in names
    assert "ld_library_path" in names
    assert "cc1plus" in names
    assert not report.ok


def test_preflight_passes_with_complete_env(tmp_path):
    script = tmp_path / "run_pdf_ocr_vllm.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    model_dir = tmp_path / "DeepSeek-OCR"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-000001.safetensors").write_bytes(b"stub")
    lib_dir = tmp_path / "libjpeg"
    lib_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cc1plus = fake_bin / "cc1plus"
    cc1plus.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cc1plus.chmod(0o755)

    env = {
        "GLOSSAPI_DEEPSEEK_OCR_DEVICE": "cuda",  # force CUDA path regardless of platform
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR": "1",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB": "0",
        "GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON": sys.executable,
        "GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT": str(script),
        "GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR": str(model_dir),
        "GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH": str(lib_dir),
        "PATH": str(fake_bin),
    }
    report = check_deepseek_ocr_env(env, check_flashinfer=False)
    assert report.ok
    assert not report.errors


def test_preflight_mps_path_no_model_dir_is_info_not_error(tmp_path, monkeypatch):
    """On macOS/MPS path, missing model dir is an info (auto-download), not an error."""
    # Force MPS path and isolate from any real weights root on disk.
    monkeypatch.delenv("GLOSSAPI_WEIGHTS_ROOT", raising=False)
    env = {
        "GLOSSAPI_DEEPSEEK_OCR_DEVICE": "mps",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR": "0",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB": "1",
        # No model dir configured — should trigger auto-download info only.
    }
    report = check_deepseek_ocr_env(env, check_flashinfer=False)
    # Should have no errors specifically about model_dir / mlx_model_dir.
    error_names = {c.name for c in report.errors}
    assert "model_dir" not in error_names
    assert "mlx_model_dir" not in error_names
    # The info about auto-download should be present.
    info_names = {c.name for c in report.infos}
    assert "mlx_model_dir" in info_names


def test_preflight_mps_path_invalid_model_dir_is_error(tmp_path):
    """On MPS path, an explicitly configured MLX model dir that doesn't exist is an error."""
    env = {
        "GLOSSAPI_DEEPSEEK_OCR_DEVICE": "mps",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR": "0",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB": "1",
        "GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR": str(tmp_path / "nonexistent_mlx"),
    }
    report = check_deepseek_ocr_env(env, check_flashinfer=False)
    error_names = {c.name for c in report.errors}
    assert "mlx_model_dir" in error_names


def test_preflight_mps_path_valid_model_dir_passes(tmp_path):
    """On MPS path, a properly populated MLX model dir produces no errors."""
    mlx_dir = tmp_path / "deepseek-ocr-mlx"
    mlx_dir.mkdir()
    (mlx_dir / "config.json").write_text("{}", encoding="utf-8")
    (mlx_dir / "weights.safetensors").write_bytes(b"stub")

    env = {
        "GLOSSAPI_DEEPSEEK_OCR_DEVICE": "mps",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR": "1",
        "GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB": "0",
        "GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR": str(mlx_dir),
    }
    report = check_deepseek_ocr_env(env, check_flashinfer=False)
    # vLLM-specific checks (cc1plus, ld_library_path, etc.) must NOT appear as errors.
    error_names = {c.name for c in report.errors}
    assert "mlx_model_contents" not in error_names
    assert "cc1plus" not in error_names
    assert "ld_library_path" not in error_names
    assert "vllm_script" not in error_names
