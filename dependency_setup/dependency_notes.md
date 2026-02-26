# GlossAPI Dependency Profiles & Test Notes

## Environment Profiles
- **Vanilla** – core GlossAPI pipeline without GPU OCR add-ons. Uses `dependency_setup/base/requirements-glossapi-vanilla.txt`.
- **RapidOCR** – Docling + RapidOCR GPU stack. Builds on vanilla requirements and adds ONNX runtime (`dependency_setup/base/requirements-glossapi-rapidocr.txt`). On macOS, the installer switches to `dependency_setup/macos/requirements-glossapi-rapidocr-macos.txt` with `onnxruntime==1.18.1`.
- **DeepSeek-OCR** – GPU OCR via DeepSeek. Two hardware paths are available:
  - *CUDA/vLLM (Linux/Windows):* Extends vanilla requirements with torch/cu128, nightly vLLM and supporting CUDA libs (`dependency_setup/base/requirements-glossapi-deepseek-ocr.txt`). `xformers` was dropped because the published wheels still pin Torch 2.8; the rest of the stack now installs cleanly on Torch 2.9.
  - *MPS/MLX (macOS Apple Silicon):* Install the `deepseek-ocr-mlx` extra (`pip install '.[deepseek-ocr-mlx]'`). No CUDA or vLLM required. Model (`mlx-community/DeepSeek-OCR-8bit`) is auto-downloaded from HuggingFace or cached under `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-1-mlx/`.
- **DeepSeek OCR v2** – MLX/MPS OCR on macOS. Uses `dependency_setup/macos/requirements-glossapi-deepseek-ocr-2-macos.txt` (vanilla + `mlx`).
- **MinerU** – OCR via the external `magic-pdf` CLI. Uses `dependency_setup/base/requirements-glossapi-mineru.txt` (vanilla + CLI setup).
- **GLM-OCR** – Lightweight 0.5B VLM OCR on macOS Apple Silicon via MLX. Uses `dependency_setup/base/requirements-glossapi-glm-ocr.txt` (vanilla + `mlx-lm`, `mlx-vlm`).
- **OlmOCR** – High-accuracy VLM-based OCR. Uses `dependency_setup/base/requirements-glossapi-olmocr.txt` (vanilla + vLLM/MLX).

Each profile is installed through `dependency_setup/setup_glossapi.sh`:
```bash
# Examples (venv path optional)
./dependency_setup/setup_glossapi.sh --mode vanilla  --venv dependency_setup/.venvs/vanilla  --run-tests
./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests
./dependency_setup/setup_glossapi.sh --mode deepseek-ocr --venv dependency_setup/.venvs/deepseek-ocr --run-tests
./dependency_setup/setup_glossapi.sh --mode deepseek-ocr-2 --venv dependency_setup/.venvs/deepseek-ocr-2 --run-tests
./dependency_setup/setup_glossapi.sh --mode mineru  --venv dependency_setup/.venvs/mineru  --run-tests
./dependency_setup/setup_glossapi.sh --mode glm-ocr --venv dependency_setup/.venvs/glm-ocr --run-tests
./dependency_setup/setup_glossapi.sh --mode olmocr  --venv dependency_setup/.venvs/olmocr  --run-tests
```

Key flags:
- `--download-deepseek-ocr` optionally fetches DeepSeek-OCR weights (skipped by default; set `--weights-root` to control where they are stored).
- `--smoke-test` (DeepSeek-OCR only) runs `dependency_setup/deepseek_ocr_gpu_smoke.py`.

## Test Segmentation
Pytest markers were added so suites can be run per profile:
- `rapidocr` – GPU Docling/RapidOCR integration tests.
- `deepseek_ocr` – DeepSeek-OCR execution paths.
- Unmarked tests cover the vanilla footprint.

`setup_glossapi.sh` now chooses marker expressions automatically:

| Mode      | Command run by script                                   |
|-----------|---------------------------------------------------------|
| vanilla   | `pytest -q -m "not rapidocr and not deepseek_ocr" tests`    |
| rapidocr  | `pytest -q -m "not deepseek_ocr" tests`                     |
| deepseek-ocr  | `pytest -q -m "not rapidocr" tests`                     |
| deepseek-ocr-2 | `pytest -q -m "not rapidocr and not deepseek_ocr" tests` |
| glm-ocr   | `pytest -q -m "not rapidocr and not deepseek_ocr" tests` |
| olmocr    | `pytest -q -m "not rapidocr and not deepseek_ocr" tests` |

Heavy GPU tests in `tests/test_pipeline_smoke.py` were guarded with `pytest.importorskip("onnxruntime")` so vanilla installs skip them cleanly. Helper PDFs now embed DejaVuSans with Unicode support and insert spacing to keep OCR-friendly glyphs.

## Validation Runs (2025-10-30)
- `./dependency_setup/setup_glossapi.sh --mode vanilla  --venv dependency_setup/.venvs/vanilla  --run-tests`
- `./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests`
- `./dependency_setup/setup_glossapi.sh --mode deepseek-ocr --venv dependency_setup/.venvs/deepseek-ocr --run-tests`

All three completed successfully after the following adjustments:
1. **Rust extensions** – switched to `pip install -e rust/glossapi_rs_{cleaner,noise}` because `maturin develop` left the wheel unregistered.
2. **Parquet locking** – `_parquet_lock` now creates parent directories before attempting the file lock (fixes `FileNotFoundError` in concurrent metadata tests).
3. **RapidOCR pipeline** – fixed `GlossExtract.create_extractor()` to build the Docling converter regardless of import path and added UTF-8 PDF generation improvements; smoke tests now pass on CUDA.
4. **DeepSeek-OCR stack** – updated nightly vLLM pin (`0.11.1rc5.dev58+g60f76baa6.cu129`) and removed `xformers` to resolve Torch 2.9 dependency conflicts.

## Known Follow-ups
- **DeepSeek-OCR weights** – installer warns if weights are absent. Set `--download-deepseek-ocr` or populate `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr` before running the real CLI tests (`GLOSSAPI_RUN_DEEPSEEK_OCR_CLI=1`).
- **xformers kernels** – removed pending compatible Torch 2.9 wheels. Reintroduce once upstream publishes matching builds.
- **Patchelf warnings** – maturin emits rpath hints if `patchelf` is missing; they are benign but install `patchelf` if cleaner logs are desired.
- **Deprecation noise** – Docling emits future warnings (Pydantic) and RapidOCR font deprecation notices; currently harmless but worth tracking for future upgrades.

## Quick Reference
- Activate an environment: `source dependency_setup/.venvs/<mode>/bin/activate`
- Re-run tests manually:
  - Vanilla: `pytest -m "not rapidocr and not deepseek_ocr" tests`
  - RapidOCR: `pytest -m "not deepseek_ocr" tests`
  - DeepSeek-OCR: `pytest -m "not rapidocr" tests`
- DeepSeek-OCR runtime exports:
  ```bash
  export GLOSSAPI_DEEPSEEK_OCR_PYTHON="dependency_setup/.venvs/deepseek-ocr/bin/python"
  export GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT="/mnt/data/glossAPI/deepseek-ocr/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH="/mnt/data/glossAPI/deepseek-ocr/libjpeg-turbo/lib"
  export LD_LIBRARY_PATH="$GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
  ```

- DeepSeek OCR v2 runtime exports (macOS MLX):
  ```bash
  export GLOSSAPI_DEEPSEEK2_PYTHON="dependency_setup/.venvs/deepseek-ocr-2/bin/python"
  export GLOSSAPI_DEEPSEEK2_MLX_SCRIPT="/mnt/data/glossAPI/deepseek-ocr-2/run_pdf_ocr_mlx.py"
  export GLOSSAPI_DEEPSEEK2_MODEL_DIR="/mnt/data/glossAPI/deepseek-ocr-2/DeepSeek-OCR-MLX"
  export GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0
  export GLOSSAPI_DEEPSEEK2_ENABLE_OCR=1
  export GLOSSAPI_DEEPSEEK2_DEVICE="mps"
  ```

- GLM-OCR runtime exports (macOS MLX):
  ```bash
  export GLOSSAPI_GLMOCR_PYTHON="dependency_setup/.venvs/glm-ocr/bin/python"
  export GLOSSAPI_GLMOCR_MLX_SCRIPT="/path/to/glossAPI/src/glossapi/ocr/glm_ocr/mlx_cli.py"
  export GLOSSAPI_GLMOCR_MODEL_DIR="/path/to/model_weights/glm-ocr-mlx"
  export GLOSSAPI_GLMOCR_ENABLE_STUB=0
  export GLOSSAPI_GLMOCR_ENABLE_OCR=1
  export GLOSSAPI_GLMOCR_DEVICE="mps"
  ```

These notes capture the current dependency state, the rationale behind constraint changes, and the validation steps used to exercise each profile.
