# Configuration & Environment Variables

This page lists the main knobs you can use to tune GlossAPI.

## GPU & Providers

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling (inside a worker), e.g. `cuda:0`.
- `GLOSSAPI_GPU_BATCH_SIZE`: batch size for GPU extraction workers (multi-GPU mode).
- macOS Apple Silicon (Metal/MPS): use `accel_type='MPS'` or set `GLOSSAPI_DOCLING_DEVICE=mps` for the RapidOCR/Docling path.

## Model Weights

All model weights are stored under a single root directory controlled by `GLOSSAPI_WEIGHTS_ROOT`:

- `GLOSSAPI_WEIGHTS_ROOT`: root directory for all model weights (default: `<repo>/model_weights`). Each backend stores weights in a subdirectory:

```
model_weights/              # $GLOSSAPI_WEIGHTS_ROOT
├── deepseek-ocr/           # DeepSeek OCR V1 CUDA/vLLM weights
├── deepseek-ocr-1-mlx/     # DeepSeek OCR V1 MLX weights (macOS Apple Silicon)
├── deepseek-ocr-mlx/       # DeepSeek OCR v2 MLX weights
├── glm-ocr-mlx/            # GLM-OCR MLX weights
├── olmocr/                 # OlmOCR-2 CUDA/vLLM weights
├── olmocr-mlx/             # OlmOCR-2 MLX weights (macOS Apple Silicon)
└── mineru/                 # MinerU PDF-Extract-Kit models
```

Individual backend directories can be overridden with their respective env vars (e.g. `GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR`, `GLOSSAPI_DEEPSEEK2_MODEL_DIR`, `GLOSSAPI_GLMOCR_MODEL_DIR`, `GLOSSAPI_OLMOCR_MLX_MODEL_DIR`) for advanced use cases. When set, they take precedence over the weights root convention. In most setups only `GLOSSAPI_WEIGHTS_ROOT` is needed.

## OCR & Parsing

- `GLOSSAPI_IMAGES_SCALE`: image scale hint for OCR/layout (default ~1.1–1.25).
- `GLOSSAPI_FORMULA_BATCH`: inline CodeFormula batch size (default `16`).
- `GLOSSAPI_OCR_LANGS`: override OCR language list (comma-separated).
- `GLOSSAPI_SKIP_DOCLING_BOOT`: set to `1` to skip Docling/RapidOCR patching at import time (useful in environments without Docling).

### Batch Policy & PDF Backend

GlossAPI exposes two Phase‑1 profiles. Use `Corpus.extract(..., phase1_backend='docling')` to switch from the default safe backend. The legacy environment variables `GLOSSAPI_BATCH_POLICY` and `GLOSSAPI_BATCH_MAX` are still parsed for backward compatibility but emit a deprecation warning and will be removed in a future release.

Regardless of backend, the extractor clamps OMP/OpenBLAS/MKL pools to one thread per worker so multi‑GPU runs do not explode thread counts.

### DeepSeek-OCR optional dependencies

Install the CUDA/vLLM extras to enable the DeepSeek-OCR path on Linux with an
NVIDIA GPU.  Imports remain lazy, so the package itself is always optional.

```bash
pip install '.[deepseek-ocr]'

# Install Torch CUDA 12.1 wheels (required by the DeepSeek-OCR script)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'

# Alternatively, use the requirements file (edit to uncomment torch lines):
pip install -r deepseek-ocr/requirements-deepseek-ocr.txt
```

**Apple Silicon / MPS path** — on macOS, DeepSeek-OCR (V1) runs via
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) without vLLM.  No CUDA wheel is
needed.  Install the MPS extras instead:

```bash
pip install '.[deepseek-ocr-mlx]'
```

The model (`mlx-community/DeepSeek-OCR-8bit`) is downloaded automatically on
first use.  To keep it locally, set `GLOSSAPI_WEIGHTS_ROOT` pointing at your
weights directory; the model will be cached under
`<weights_root>/deepseek-ocr-1-mlx/`.

When using `backend='deepseek-ocr'`, equations are included inline in the OCR output; Phase‑2 math flags are accepted but skipped.

When using `backend='deepseek-ocr-2'`, equations are included inline in the OCR output; Phase-2 math flags are accepted but skipped.

When using `backend='mineru'`, equations are included inline in the OCR output; Phase‑2 math flags are accepted but skipped.

### DeepSeek-OCR runtime controls

The runner auto-detects the platform and selects the right execution path:

| Platform | Strategies tried (in order) |
|---|---|
| macOS (Apple Silicon) | In-process MLX → MLX CLI subprocess → stub |
| Linux / Windows (CUDA) | vLLM CLI subprocess → stub |

**Common controls:**

- `GLOSSAPI_DEEPSEEK_OCR_ALLOW_STUB` (`1` by default): allow the builtin stub runner for tests and lightweight environments.
- `GLOSSAPI_DEEPSEEK_OCR_DEVICE`: force device selection — `mps`, `cuda`, or `cpu`.  Auto-detected when unset (`mps` on macOS, `cuda` elsewhere).

**MPS / MLX controls (Apple Silicon):**

- `GLOSSAPI_DEEPSEEK_OCR_ALLOW_MLX_CLI` (`1` by default on the MPS path): set to `0` to skip the MLX CLI subprocess strategy, or `1` to force it even when the kwarg says otherwise.
- `GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR`: local directory containing MLX-formatted weights and `config.json`. If unset, weights are resolved from `GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-1-mlx/` or auto-downloaded from `mlx-community/DeepSeek-OCR-8bit`.
- `GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL`: HuggingFace model ID for auto-download (default `mlx-community/DeepSeek-OCR-8bit`).
- `GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT`: override path to the MLX CLI inference script. By default the package-shipped `mlx_cli.py` is used.
- `GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON` / `GLOSSAPI_DEEPSEEK_OCR_PYTHON`: Python executable for the MLX CLI subprocess (defaults to the current interpreter).

**CUDA / vLLM controls (Linux):**

- `GLOSSAPI_DEEPSEEK_OCR_ALLOW_CLI` (`0` by default): flip to `1` to force the real vLLM CLI even when the stub is allowed.
- `GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT`: override path to the DeepSeek-OCR vLLM CLI script (defaults to `deepseek-ocr/run_pdf_ocr_vllm.py` under the repo).
- `GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR`: path to CUDA model weights (must contain `config.json` + safetensors).
- `GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH`: prepend extra library search paths (e.g., for `libjpeg-turbo`) when launching the CLI.
- `GLOSSAPI_DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION`: VRAM fraction (0.5–0.9) for vLLM.
- `GLOSSAPI_DEEPSEEK_OCR_NO_FP8_KV`: set to `1` to disable FP8 KV cache (propagates `--no-fp8-kv`).

### DeepSeek OCR v2 (MLX/MPS) runtime controls

The runner tries three strategies in order: **in-process** (fast, model stays loaded), **CLI subprocess**, then **stub**.

- `GLOSSAPI_DEEPSEEK2_ALLOW_STUB` (`1` by default): allow the builtin stub runner for tests and lightweight environments.
- `GLOSSAPI_DEEPSEEK2_ALLOW_CLI` (`0` by default): flip to `1` to force the CLI subprocess strategy.
- `GLOSSAPI_DEEPSEEK2_MODEL_DIR`: model directory containing MLX-formatted weights and config. If unset, models are auto-downloaded from `mlx-community/DeepSeek-OCR-2-8bit` on HuggingFace.
- `GLOSSAPI_DEEPSEEK2_DEVICE`: device override (`mps` or `cpu`, default `mps`).
- `GLOSSAPI_DEEPSEEK2_MLX_SCRIPT`: override path to an external MLX CLI script. Only needed when using a separate venv or custom script. By default the package-shipped script is used.
- `GLOSSAPI_DEEPSEEK2_PYTHON` / `GLOSSAPI_DEEPSEEK2_TEST_PYTHON`: absolute path to the Python interpreter for CLI subprocess mode (defaults to the current interpreter).

### MinerU runtime controls

- `GLOSSAPI_MINERU_ALLOW_STUB` (`1` by default): allow the builtin stub runner for tests and lightweight environments.
- `GLOSSAPI_MINERU_ALLOW_CLI` (`0` by default): flip to `1` to run `magic-pdf` when available.
- `GLOSSAPI_MINERU_COMMAND`: override the `magic-pdf` executable path.
- `GLOSSAPI_MINERU_MODE`: override the MinerU mode flag (passed to `magic-pdf -m`, default `auto`).
- `GLOSSAPI_MINERU_BACKEND`: override the MinerU backend (passed to `magic-pdf -b`, e.g. `pipeline`, `hybrid-auto-engine`, `vlm`).
- `GLOSSAPI_MINERU_DEVICE_MODE`: override the MinerU device mode (`mps`, `cuda`, or `cpu`). Requires `MINERU_TOOLS_CONFIG_JSON` to point at the base config.

#### MinerU doctor checks

Run the preflight checker to validate your CLI, config, device, and model paths:

```bash
python -m glossapi.ocr.mineru.preflight
```

### GLM-OCR runtime controls

GLM-OCR is a compact 0.5B VLM for document OCR, running on Apple Silicon via MLX.

- `GLOSSAPI_GLMOCR_ALLOW_STUB` (`1` by default): allow the builtin stub runner for tests and lightweight environments.
- `GLOSSAPI_GLMOCR_ALLOW_CLI` (`0` by default): flip to `1` to run the GLM-OCR MLX CLI subprocess.
- `GLOSSAPI_GLMOCR_PYTHON`: Python executable for the GLM-OCR venv.
- `GLOSSAPI_GLMOCR_MODEL_DIR`: local model weights directory (takes precedence over HF model ID).
- `GLOSSAPI_GLMOCR_MLX_MODEL`: HuggingFace MLX model identifier (default `mlx-community/GLM-OCR-4bit`).
- `GLOSSAPI_GLMOCR_MLX_SCRIPT`: path to the MLX CLI inference script for subprocess execution.
- `GLOSSAPI_GLMOCR_DEVICE`: device override (`mps`, `cpu`); auto-detected if unset.

On macOS, the runner tries in-process MLX first (if `mlx_vlm` is importable), then
the MLX CLI subprocess, then the stub.

#### GLM-OCR doctor checks

Run the preflight checker to validate your environment:

```bash
python -m glossapi.ocr.glm_ocr.preflight
```

### OlmOCR-2 runtime controls

#### CUDA / vLLM

- `GLOSSAPI_OLMOCR_ALLOW_STUB` (`1` by default): allow the builtin stub runner for tests and lightweight environments.
- `GLOSSAPI_OLMOCR_ALLOW_CLI` (`0` by default): flip to `1` to run the real OlmOCR pipeline (external `olmocr` package).
- `GLOSSAPI_OLMOCR_PYTHON`: Python executable for the OlmOCR venv.
- `GLOSSAPI_OLMOCR_MODEL`: HuggingFace model identifier (default `allenai/olmOCR-2-7B-1025-FP8`).
- `GLOSSAPI_OLMOCR_MODEL_DIR`: local CUDA model weights directory (takes precedence over HF model ID).
- `GLOSSAPI_OLMOCR_SERVER`: URL of an external vLLM server (skips spawning a local vLLM instance).
- `GLOSSAPI_OLMOCR_API_KEY`: API key for external vLLM server.
- `GLOSSAPI_OLMOCR_GPU_MEMORY_UTILIZATION`: fraction of VRAM vLLM may pre-allocate for KV-cache (default `0.85`).
- `GLOSSAPI_OLMOCR_MAX_MODEL_LEN`: upper bound (tokens) vLLM will allocate KV-cache for (default `8192`).
- `GLOSSAPI_OLMOCR_TENSOR_PARALLEL_SIZE`: tensor parallel size for vLLM (default `1`).
- `GLOSSAPI_OLMOCR_TARGET_IMAGE_DIM`: dimension on longest side used for rendering PDF pages (OlmOCR CLI only).
- `GLOSSAPI_OLMOCR_WORKERS`: number of OlmOCR pipeline workers (OlmOCR CLI only).
- `GLOSSAPI_OLMOCR_PAGES_PER_GROUP`: number of PDF pages per work item group (OlmOCR CLI only).
- `GLOSSAPI_OLMOCR_VLLM_SCRIPT`: path to the vLLM CLI inference script for subprocess execution (default: package-embedded `vllm_cli.py`).
- `GLOSSAPI_OLMOCR_LD_LIBRARY_PATH`: extra library paths prepended to `LD_LIBRARY_PATH` for vLLM/OlmOCR CLI subprocesses (e.g. `/usr/local/cuda/lib64`). Fixes `libcudart.so.12 not found` errors when the CUDA runtime lives outside the default search path.

#### MPS / MLX (macOS Apple Silicon)

- `GLOSSAPI_OLMOCR_MLX_MODEL`: HuggingFace MLX model identifier (default `mlx-community/olmOCR-2-7B-1025-4bit`).
- `GLOSSAPI_OLMOCR_MLX_MODEL_DIR`: local MLX-formatted model weights directory.
- `GLOSSAPI_OLMOCR_MLX_SCRIPT`: path to the MLX CLI inference script for subprocess execution.
- `GLOSSAPI_OLMOCR_DEVICE`: device override (`cuda`, `mps`, `cpu`); auto-detected if unset.

#### Strategy cascade

The runner tries strategies in this order:

1. **In-process MLX** (macOS only, if `mlx_vlm` is importable)
2. **MLX CLI subprocess** (macOS only)
3. **In-process vLLM** (Linux/CUDA only, if `vllm` is importable and CUDA is available)
4. **vLLM CLI subprocess** (Linux/CUDA only)
5. **OlmOCR CLI subprocess** (requires `olmocr` package and `GLOSSAPI_OLMOCR_ALLOW_CLI=1`)
6. **Stub** (default fallback for testing)

On macOS, strategies 3–4 are skipped. On Linux, strategies 1–2 are skipped.

#### OlmOCR doctor checks

Run the preflight checker to validate your environment:

```bash
python -m glossapi.ocr.olmocr.preflight
```

## Math Enrichment (Phase‑2)

- `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable/disable early‑stop wrapper.
- `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`): cap decoded LaTeX length.
- `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`): stop on last‑token repetition runs.
- `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap decoder new tokens.
- `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`): stride for length checks.
- `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`): maximum number of times a crashed math worker is respawned per GPU during multi‑GPU enrichment (set to `0` to disable respawns).

### Centralized LaTeX Policy (Post‑processing)

- `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default `1`): only sanitize when output looks problematic.
- `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default `50`): consider output failed if tail token repeats more than this.
- `GLOSSAPI_LATEX_POST_WINDDOWN` (default `12`): clamp repeated tail token to this run length.
- `GLOSSAPI_LATEX_POST_MAX_CHARS` (default `3000`): cap text length (prefers whitespace/`\` boundary).

All LaTeX policy knobs are loaded via `glossapi.text_sanitize.load_latex_policy()` and used consistently in early‑stop, Phase‑2 post‑processing, and metrics.

## Performance & Caches

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription.
- Cache locations: `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`.

## Worker Logging

- `GLOSSAPI_WORKER_LOG_DIR`: override the directory used for per-worker logs and `gpu<N>.current` markers (defaults to `logs/ocr_workers/` or `logs/math_workers/` under the output directory).
- `GLOSSAPI_WORKER_LOG_VERBOSE` = `1|0` (default `1`): emit (or suppress) the GPU binding banner each worker prints on startup.

## RapidOCR Model Paths

- `GLOSSAPI_RAPIDOCR_ONNX_DIR`: directory containing `det/rec/cls` ONNX models and keys.

## Triage & Parquet

- Triage always writes both:
  - Sidecar summaries: `sidecars/triage/{stem}.json` (per document)
  - Parquet updates: `download_results/download_results.parquet` (adds/updates rows)
- Default recommendation policy: enrich if `formula_total > 0` (skip only no‑math docs).
- Legacy heuristic (p90/pages thresholds) can be enabled with `GLOSSAPI_TRIAGE_HEURISTIC=1`.

## Skiplist

- `GLOSSAPI_SKIPLIST_PATH`: override the path to the fatal skip-list file (defaults to `<output_dir>/skiplists/fatal_skip.txt`).
- Files on the skip-list are never retried. Remove entries manually or use `reprocess_completed=True` to force reprocessing of non-fatal stems.
