# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.8+ (3.11–3.12 recommended; CI uses 3.11)
- Recent `pip` (or `uv`) and a C/C++ toolchain for Rust wheels
- Rust stable toolchain (for building `glossapi_rs_cleaner` and `glossapi_rs_noise`)
- Optional: NVIDIA GPU with CUDA 12.x drivers for Docling/RapidOCR acceleration
- Optional: Apple Silicon Mac with Metal/MPS for GPU-accelerated OCR on macOS

## Install GlossAPI

### Recommended — mode-aware setup script

Use `dependency_setup/setup_glossapi.sh` to build an isolated virtualenv with the correct dependency set. Supported modes: `vanilla`, `rapidocr`, `deepseek-ocr`, `deepseek-ocr-2`, `glm-ocr`, `olmocr`, `mineru`. Examples:

```bash
# Vanilla pipeline (CPU-only OCR)
./dependency_setup/setup_glossapi.sh --mode vanilla --venv dependency_setup/.venvs/vanilla --run-tests

# RapidOCR GPU stack
./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests

# DeepSeek-OCR on GPU (weights stored under $GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr)
./dependency_setup/setup_glossapi.sh \
  --mode deepseek-ocr \
  --venv dependency_setup/.venvs/deepseek-ocr \
  --weights-root /path/to/model_weights \
  --run-tests --smoke-test
```

Add `--download-deepseek-ocr` if you need the script to fetch weights via Hugging Face; otherwise set `GLOSSAPI_WEIGHTS_ROOT` so the pipeline finds weights at `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr`. Inspect `dependency_setup/dependency_notes.md` for the latest pins, caveats, and validation runs. The script auto-detects Python (preferring 3.12 → 3.11 → 3.13) and installs GlossAPI with its Rust crates in editable mode so source changes are picked up immediately.

### DeepSeek OCR v2 (MLX/MPS) setup

```bash
./dependency_setup/setup_glossapi.sh \
  --mode deepseek-ocr-2 \
  --venv dependency_setup/.venvs/deepseek-ocr-2 \
  --download-deepseek-ocr2 \
  --run-tests
```

- Targets macOS Apple Silicon via MLX.
- Weights are auto-downloaded from `mlx-community/DeepSeek-OCR-2-8bit` if `--download-deepseek-ocr2` is passed.
- Equations are included inline — Phase-2 math enrichment is a no-op.

**DeepSeek OCR v2 runtime checklist**
- Run `python -m glossapi.ocr.deepseek_ocr2.preflight` to validate the environment.
- The runner tries in-process MLX first (fastest), then CLI subprocess, then stub.
- Set `GLOSSAPI_DEEPSEEK2_ALLOW_STUB=0` to force real OCR.
- Override model path with `GLOSSAPI_DEEPSEEK2_MODEL_DIR` if needed.
- Override device with `GLOSSAPI_DEEPSEEK2_DEVICE` (`mps` or `cpu`, default `mps`).

### GLM-OCR setup

```bash
./dependency_setup/setup_glossapi.sh \
  --mode glm-ocr \
  --venv dependency_setup/.venvs/glm-ocr \
  --download-glmocr \
  --run-tests
```

- Targets macOS Apple Silicon via MLX.
- Weights are auto-downloaded from `mlx-community/GLM-OCR-4bit` if `--download-glmocr` is passed.
- Equations are included inline — Phase-2 math enrichment is a no-op.

**GLM-OCR runtime checklist**
- Run `python -m glossapi.ocr.glm_ocr.preflight` to validate the environment.
- The runner tries in-process MLX first (fastest), then CLI subprocess, then stub.
- Set `GLOSSAPI_GLMOCR_ALLOW_STUB=0` to force real OCR.
- Override model path with `GLOSSAPI_GLMOCR_MODEL_DIR` if needed.
- Override device with `GLOSSAPI_GLMOCR_DEVICE` (`mps` or `cpu`, default `mps`).

### MinerU setup

```bash
./dependency_setup/setup_glossapi.sh \
  --mode mineru \
  --venv dependency_setup/.venvs/mineru \
  --download-mineru-models \
  --run-tests
```

- Use Python 3.11 for the MinerU venv (3.10–3.13 supported upstream).
- Ensure `magic-pdf` is on PATH (or set `GLOSSAPI_MINERU_COMMAND`).
- Equations are included inline — Phase-2 math enrichment is a no-op.

**MinerU runtime checklist**
- Run `python -m glossapi.ocr.mineru.preflight` to validate CLI, config, device, and model paths.
- Set `GLOSSAPI_MINERU_ALLOW_CLI=1` and `GLOSSAPI_MINERU_ALLOW_STUB=0` to force real OCR.
- For macOS GPU: set `GLOSSAPI_MINERU_BACKEND="hybrid-auto-engine"` and `GLOSSAPI_MINERU_DEVICE_MODE="mps"`.

**DeepSeek-OCR runtime checklist**
- Run `python -m glossapi.ocr.deepseek_ocr.preflight` from the DeepSeek-OCR venv to assert the CLI can run (env vars, model dir, flashinfer, cc1plus, libjpeg).
- Force the real CLI and avoid stub fallback by setting:
  - `GLOSSAPI_DEEPSEEK_OCR_ALLOW_CLI=1`
  - `GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB=0`
  - `GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT=/path/to/deepseek-ocr/run_pdf_ocr_vllm.py`
  - `GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON=/path/to/deepseek-ocr/venv/bin/python`
  - `GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR=/path/to/deepseek-ocr/DeepSeek-OCR`
  - `GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib`
- Install a CUDA toolkit with `nvcc` and set `CUDA_HOME` / prepend `$CUDA_HOME/bin` to `PATH` (FlashInfer/vLLM JIT expects it).
- If FlashInfer is unstable on your stack, disable it with `VLLM_USE_FLASHINFER=0` and `FLASHINFER_DISABLE=1`.
- Avoid FP8 KV cache issues by exporting `GLOSSAPI_DEEPSEEK_OCR_NO_FP8_KV=1`; tune VRAM use via `GLOSSAPI_DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION=<0.5–0.9>`.
- Keep `LD_LIBRARY_PATH` pointing at the toolkit lib64 (e.g. `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`).

### Option 1 — pip (evaluate quickly)

```bash
export PYTHONNOUSERSITE=1  # keep ~/.local packages out of the way
pip install glossapi
```

### Option 2 — local development (recommended)

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python -m venv .venv && source .venv/bin/activate
pip install -U pip maturin
pip install -e .
```

This builds the Rust extensions needed for `Corpus.clean()` and noise metrics. Re-run `pip install -e .` after pulling changes that touch Rust crates.

### Option 3 — conda on SageMaker / Amazon Linux

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
conda activate glossapi
```

The helper script provisions Python 3.10, installs Rust + `maturin`, performs an editable install, and applies the Docling RapidOCR patch automatically.

## GPU prerequisites (optional but recommended)

`setup_glossapi.sh` pulls the right CUDA/Torch/ONNX wheels for the RapidOCR and DeepSeek-OCR profiles. If you are curating dependencies manually, make sure you:

- Install the GPU build of ONNX Runtime (`onnxruntime-gpu`) and uninstall the CPU wheel.
- Select the PyTorch build that matches your driver/toolkit (the repository currently targets CUDA 12.1 for DeepSeek-OCR).
- Verify the providers with:

  ```bash
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## RapidOCR models & keys

GlossAPI ships the required ONNX models and Greek keys under `glossapi/models/rapidocr/{onnx,keys}`. To override them, set `GLOSSAPI_RAPIDOCR_ONNX_DIR` to a directory containing:

- `det/inference.onnx`
- `rec/inference.onnx`
- `cls/ch_ppocr_mobile_v2.0_cls_infer.onnx`
- `greek_ppocrv5_keys.txt`

## First run (lightweight corpus)

```bash
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")
PY
```

- Inspect `artifacts/lightweight_pdf_run/markdown/` and compare with `samples/lightweight_pdf_corpus/expected_outputs.json`.
- Run `pytest tests/test_pipeline_smoke.py` for a reproducible regression check tied to the same corpus.

## Interactive CLI

Once installed, GlossAPI offers a Typer-based CLI with interactive wizards:

```bash
glossapi              # Launches pipeline wizard (default)
glossapi pipeline     # Interactive phase-selection wizard with gum
glossapi setup        # Environment provisioning wizard
```

Both wizards use [gum](https://github.com/charmbracelet/gum) for rich prompts, falling back to simple TTY prompts if gum is unavailable.

## Next steps

- Jump into [Quickstart recipes](quickstart.md) for GPU OCR, Docling, and enrichment commands.
- Explore [Pipeline overview](pipeline.md) to understand each processing stage and emitted artifact.
- Read the [Corpus API reference](api_corpus_tmp.md) for complete method signatures.
- See [Configuration](configuration.md) for the full list of `GLOSSAPI_*` environment variables.
- When ready to contribute docs, expand the placeholders in `docs/divio/`.
