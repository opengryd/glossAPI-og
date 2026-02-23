# GPU OCR and Math Enrichment

This document summarizes how GlossAPI uses the GPU for OCR and formula/code enrichment, how to run each phase efficiently, and where artifacts are written.

## Overview

- Phase‑1 (Extract): PDF → Markdown via Docling; optional GPU OCR via RapidOCR (ONNXRuntime). Optionally emit JSON + formula index for Phase‑2.
- Phase‑2 (Enrich): From Docling JSON, decode math/code on the GPU (CodeFormula) and re‑emit enriched Markdown.

Backends

| Backend | GPU platform | Math handling |
|---|---|---|
| `backend='rapidocr'` (default) | CUDA (Linux/Windows), CoreML/MPS (macOS) | Separate Phase-2 enrichment from Docling JSON |
| `backend='deepseek-ocr'` | CUDA/vLLM (Linux/Windows only) | Inline — Phase-2 is a no-op |
| `backend='deepseek-ocr-2'` | MLX/MPS (macOS Apple Silicon only) | Inline — Phase-2 is a no-op |
| `backend='glm-ocr'` | MLX/MPS (macOS Apple Silicon only) | Inline — Phase-2 is a no-op |
| `backend='olmocr'` | CUDA/vLLM (Linux) **or** MLX/MPS (macOS Apple Silicon) | Inline — Phase-2 is a no-op |
| `backend='mineru'` | CUDA / MPS / CPU | Inline — Phase-2 is a no-op |

Policy: never OCR and math on the same file
- If a file needs OCR, GlossAPI runs OCR only (no Phase‑2 on that file in the same pass).
- If a file does not need OCR but needs math, GlossAPI runs math‑only from Docling JSON. The JSON is produced by Phase‑1 (Docling layout) and must already exist.

### Python API layout

- DeepSeek-OCR entry point: `glossapi.ocr.deepseek_ocr.runner.run_for_files(...)`
- DeepSeek OCR v2 entry point: `glossapi.ocr.deepseek_ocr2.runner.run_for_files(...)`
- MinerU entry point: `glossapi.ocr.mineru.runner.run_for_files(...)`
- GLM-OCR entry point: `glossapi.ocr.glm_ocr.runner.run_for_files(...)`
- OlmOCR-2 entry point: `glossapi.ocr.olmocr.runner.run_for_files(...)`
- RapidOCR dispatcher: `glossapi.ocr.rapidocr.dispatch.run_via_extract(...)`
- Math enrichment: `glossapi.ocr.math.enrich.enrich_from_docling_json(...)`
- Utility helpers (Docling JSON / cleaning): `glossapi.ocr.utils.*`

## Prerequisites

### RapidOCR / Docling (CUDA or CoreML/MPS)

- Install: `pip install '.[rapidocr]'`
- Linux/Windows GPU: `onnxruntime-gpu==1.18.1` (uninstall `onnxruntime` CPU first) + `torch==2.5.1+cu121`
- macOS (Apple Silicon): `onnxruntime==1.18.1` with CoreMLExecutionProvider (Metal) + PyTorch with MPS support
- Packaged RapidOCR models/keys under `glossapi/models/rapidocr/{onnx,keys}` (or `GLOSSAPI_RAPIDOCR_ONNX_DIR`)
- Optional for Phase-2 JSON: `pypdfium2`, `zstandard`

### DeepSeek-OCR (CUDA/vLLM — Linux/Windows only)

- Install: `pip install '.[deepseek-ocr]'` in a dedicated venv
- Requires: NVIDIA GPU, CUDA 12.x toolkit with `nvcc`, `torch==2.5.1+cu121`, `vllm`
- Not supported on macOS (use `deepseek-ocr-2` or `glm-ocr` on Apple Silicon instead)

### DeepSeek OCR v2 (MLX — macOS Apple Silicon only)

- Install: `pip install mlx-vlm pypdfium2 Pillow` in a dedicated venv
- Requires: Apple Silicon Mac (M1+); will not work on Intel Macs or Linux
- Weights: auto-downloaded from `mlx-community/DeepSeek-OCR-2-8bit` or set `GLOSSAPI_DEEPSEEK2_MODEL_DIR`

### GLM-OCR (MLX — macOS Apple Silicon only)

- Install: `pip install 'mlx-vlm>=0.3.12' pypdfium2 Pillow`
- Requires: Apple Silicon Mac (M1+); will not work on Intel Macs or Linux
- Weights: auto-downloaded from `mlx-community/GLM-OCR-4bit` or set `GLOSSAPI_GLMOCR_MODEL_DIR`

### OlmOCR-2 (CUDA/vLLM on Linux **or** MLX/MPS on macOS)

- CUDA path: `pip install 'olmocr[gpu]'` in a dedicated venv; requires NVIDIA GPU with ≥12 GB VRAM and `poppler-utils` on PATH
- MLX path (macOS Apple Silicon): `pip install mlx-vlm pypdfium2 Pillow`; weights auto-downloaded from `mlx-community/olmOCR-2-7B-1025-4bit` or set `GLOSSAPI_OLMOCR_MLX_MODEL_DIR`

### MinerU (CUDA / MPS / CPU)

- Install: `pip install 'mineru[all]'` (preferably in a Python 3.11 venv)
- Ensure `magic-pdf` is on PATH (or set `GLOSSAPI_MINERU_COMMAND`)
- macOS GPU: Torch MPS is available automatically inside a `mineru[all]` venv

Verify GPU readiness before forcing OCR or math:

```bash
# Linux/Windows — CUDA (RapidOCR, DeepSeek-OCR, OlmOCR-2 CUDA path)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"  # expects True, >=1
python -c "import onnxruntime as ort; print(ort.get_available_providers())"            # must include CUDAExecutionProvider

# macOS (Apple Silicon) — MPS/Metal (RapidOCR CoreML, MinerU MPS)
python -c "import torch; print(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"            # must include CoreMLExecutionProvider

# macOS (Apple Silicon) — MLX (DeepSeek OCR v2, GLM-OCR, OlmOCR-2 MLX path)
python -c "import mlx.core as mx; print(mx.default_device())"                          # expects Device(gpu, 0)
```

## Running Phase‑1 (Extract)

```python
from glossapi import Corpus
c = Corpus('IN','OUT')

# GPU OCR on PDFs; emit JSON + formula index for Phase‑2
c.extract(
    input_format='pdf',
    accel_type='CUDA',           # Linux/Windows — or use_gpus='multi' for multi-GPU
  # accel_type='MPS',           # macOS Apple Silicon (Metal/MPS)
    force_ocr=True,              # OCR always on for PDFs
    emit_formula_index=True,     # request json/<stem>.formula_index.jsonl alongside the default JSON
)
```

When `force_ocr=True` (or when math/code enrichment is enabled), GlossAPI automatically switches to the Docling backend and aborts if CUDA/MPS-enabled torch/ONNXRuntime providers are not available.

Outputs:
- `markdown/<stem>.md`
- `json/<stem>.docling.json(.zst)` and `json/<stem>.formula_index.jsonl`
- `json/metrics/<stem>.metrics.json` and `json/metrics/<stem>.per_page.metrics.json`

## Running Phase‑2 (Enrich)

```python
from glossapi import Corpus
c = Corpus('OUT','OUT')  # same folder for both

# GPU formula/code decoding from JSON (writes enriched MD to markdown/<stem>.md)
c.formula_enrich_from_json(
    device='cuda',       # Linux/Windows — CUDA
  # device='mps',        # macOS Apple Silicon (Metal/MPS)
    batch_size=12,       # tune for your GPU
)
```

Outputs:
- `markdown/<stem>.md` — enriched Markdown overwrites the plain MD
- `json/<stem>.latex_map.jsonl` — LaTeX strings + acceptance/metrics per item

## DeepSeek-OCR usage

Run OCR for files flagged by the cleaner as needing OCR (math flags are ignored for DeepSeek-OCR):

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek-ocr', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase-2 is skipped
```

## DeepSeek OCR v2 (MLX/MPS) usage

Run OCR for files flagged by the cleaner as needing OCR (math flags are ignored for DeepSeek OCR v2):

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek-ocr-2', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase‑2 is skipped
```

The runner first tries **in-process MLX** (fast — model stays loaded across files),
then falls back to **CLI subprocess**, then **stub**.  Minimal env setup:

```bash
export GLOSSAPI_DEEPSEEK2_ALLOW_STUB=0
export GLOSSAPI_DEEPSEEK2_DEVICE=mps
# Optional: point to local weights (otherwise auto-downloaded from HuggingFace)
# export GLOSSAPI_DEEPSEEK2_MODEL_DIR=/path/to/DeepSeek-OCR-MLX
python -m glossapi.ocr.deepseek_ocr2.preflight
```


If you need Phase‑2 math on files that do not require OCR, use RapidOCR/Docling and math‑only (expects Docling JSON from Phase‑1):

```python
c.ocr(backend='rapidocr', fix_bad=False, math_enhance=True, mode='math_only')
# → runs Phase‑2 on non‑OCR files only (requires Docling JSON)
```

## MinerU usage

Run OCR for files flagged by the cleaner as needing OCR (math flags are ignored for MinerU):

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='mineru', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase‑2 is skipped
```

Recommended macOS GPU settings:

```bash
export GLOSSAPI_MINERU_BACKEND="hybrid-auto-engine"
export GLOSSAPI_MINERU_DEVICE_MODE="mps"
export MINERU_TOOLS_CONFIG_JSON="$GLOSSAPI_WEIGHTS_ROOT/mineru/magic-pdf.json"
python -m glossapi.ocr.mineru.preflight
```

Validate your MinerU setup (CLI, config, device, and model paths):

```bash
python -m glossapi.ocr.mineru.preflight
```

## OlmOCR-2 usage

OlmOCR-2 is a high-accuracy VLM-based OCR toolkit (Qwen2.5-VL fine-tune) that supports both
CUDA (via vLLM) on Linux and Apple Silicon MPS (via MLX) on macOS. Equations are included inline
— Phase-2 math enrichment is not required and is treated as a no-op.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.ocr(backend='olmocr', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase-2 is skipped
```

### macOS (MLX/MPS) minimal setup

The runner tries **in-process MLX** first (model stays loaded across files), then **MLX CLI subprocess**, then fallback strategies. Set `GLOSSAPI_OLMOCR_ALLOW_STUB=0` to prevent silent placeholder output:

```bash
export GLOSSAPI_OLMOCR_ALLOW_STUB=0
export GLOSSAPI_OLMOCR_DEVICE=mps
# Optional: point to local MLX weights (otherwise auto-downloaded from HuggingFace)
# export GLOSSAPI_OLMOCR_MLX_MODEL_DIR=/path/to/olmOCR-MLX
python -m glossapi.ocr.olmocr.preflight
```

### Linux (CUDA/vLLM) minimal setup

The runner tries **in-process vLLM** first (model stays loaded), then **vLLM CLI subprocess**, then the OlmOCR pipeline subprocess as a last resort before the stub:

```bash
export GLOSSAPI_OLMOCR_ALLOW_STUB=0
export GLOSSAPI_OLMOCR_ALLOW_CLI=1
export GLOSSAPI_OLMOCR_DEVICE=cuda
export GLOSSAPI_OLMOCR_MODEL_DIR=/path/to/model_weights/olmocr
# Tune VRAM fraction (default 0.85); lower if you see OOM:
# export GLOSSAPI_OLMOCR_GPU_MEMORY_UTILIZATION=0.75
# Fix "libcudart.so.12 not found" if CUDA runtime is not in the default library path:
# export GLOSSAPI_OLMOCR_LD_LIBRARY_PATH=/usr/local/cuda/lib64
python -m glossapi.ocr.olmocr.preflight
```

### Strategy cascade

The OlmOCR runner tries strategies in this order:

| Step | Strategy | Platform |
|---|---|---|
| 1 | In-process MLX | macOS only (`mlx_vlm` importable) |
| 2 | MLX CLI subprocess | macOS only |
| 3 | In-process vLLM | Linux/CUDA only (`vllm` importable) |
| 4 | vLLM CLI subprocess | Linux/CUDA only |
| 5 | OlmOCR CLI subprocess | Any (`GLOSSAPI_OLMOCR_ALLOW_CLI=1`) |
| 6 | Stub | Any (default; `GLOSSAPI_OLMOCR_ALLOW_STUB=1`) |

### Using an external vLLM server

For multi-node or shared-server deployments, point OlmOCR at a running vLLM endpoint:

```bash
export GLOSSAPI_OLMOCR_SERVER=http://my-vllm-host:8000
export GLOSSAPI_OLMOCR_API_KEY=sk-...   # if auth is required
```

## Multi‑GPU

Phase‑1 (extract):
```python
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```
Workers set `CUDA_VISIBLE_DEVICES` per process; Docling runs on `cuda:0` relative to each worker. OCR uses ORT GPU under the same process.

Phase‑2 (enrich):
```python
c.ocr(use_gpus='multi', math_batch_size=12)
```
Spawns math workers; each binds to its GPU using `CUDA_VISIBLE_DEVICES` and runs CodeFormula on `cuda:0` relative to that worker.

### Resuming and Recovering Workers

- `Corpus.ocr()` now persists end‑to‑end progress in the canonical parquet (`download_results/download_results.parquet`). As long as `reprocess_completed=False` (default) any row with `ocr_success=True` or `math_enriched=True` is skipped on the next run; pass `reprocess_completed=True` to force a redo or use the legacy alias `skip_existing=False`.
- Multi‑GPU math workers respawn automatically when a process crashes. Control the number of retries per GPU with `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`). Active assignments are written to `logs/math_workers/gpu<N>.current` and the worker log directory can be overridden via `GLOSSAPI_WORKER_LOG_DIR`.
- When a GPU exceeds the respawn cap the remaining stems are added to the fatal skip‑list and copied to `downloads/problematic_math/` (PDFs) and `json/problematic_math/` (JSON artifacts) so they can be inspected or retried manually.
- Set `GLOSSAPI_WORKER_LOG_VERBOSE=0` to silence the per-worker binding banner; each worker still keeps an append-only log in the worker log directory for post‑mortem debugging.

## Performance & Tuning

- Batch sizes
  - Inline (Phase‑1): `GLOSSAPI_FORMULA_BATCH` (default 16) sets CodeFormula docling side throughput.
  - Phase‑2: `batch_size` / `math_batch_size` parameter (typ. 8–16) balances VRAM and speed.
- Images scale for OCR: `GLOSSAPI_IMAGES_SCALE` (~1.1–1.25) can improve detection on thin glyphs.
- CPU threads: cap `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to avoid CPU oversubscription on multi‑GPU nodes.

## Early‑stop & Post‑Processing Guards (Formula)

To keep LaTeX well‑formed and fast:
- Generation‑time (applied inside the decoder):
  - `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable early‑stop criteria for the HF generate path.
  - `GLOSSAPI_LATEX_MAX_CHARS` (default 3000): decoded‑length stop gate.
  - `GLOSSAPI_LATEX_MAX_REPEAT` (default 50): stop on excessive last‑token repetition.
  - `GLOSSAPI_LATEX_LEN_STRIDE` (default 16): decoding stride for the length check.
  - `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap new tokens at the decoder level (injected only if caller doesn’t specify one).

- Post‑processing (applied on the generated string):
  - `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default 1): only sanitize when output looks pathological.
  - `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default 50): consider output failed if the last token repeats more than this gate.
  - `GLOSSAPI_LATEX_POST_WINDDOWN` (default 12): clamp the repeated tail token to this run length.
  - `GLOSSAPI_LATEX_POST_MAX_CHARS` (default 3000): cap text length; prefers whitespace/`\` boundary.

The policy is centralized in `glossapi.text_sanitize`. Phase‑2 enrichment and the per‑page metrics both use the same sanitizer so counts and truncation flags are consistent.

## Artifact Placement Summary

```
OUT/
├── markdown/
│   └── <stem>.md                     # enriched Markdown (canonical — overwrites Phase-1)
├── json/
│   ├── <stem>.docling.json(.zst)     # Docling layout JSON
│   ├── <stem>.formula_index.jsonl    # Formula/code item index (Phase-1)
│   ├── <stem>.latex_map.jsonl        # LaTeX strings + acceptance (Phase-2)
│   ├── metrics/
│   │   ├── <stem>.metrics.json       # Document-level metrics
│   │   └── <stem>.per_page.metrics.json  # Per-page timing + formula counts
│   └── problematic_math/            # Quarantined artifacts (respawn cap exceeded)
├── clean_markdown/                   # Rust-cleaned Markdown (from corpus.clean())
├── sidecars/
│   ├── extract/                      # Per-file extraction metadata
│   ├── triage/                       # Formula density / OCR routing decisions
│   └── math/                         # Math enrichment metadata
├── downloads/
│   └── problematic_math/            # Quarantined PDFs
└── download_results/
    └── download_results.parquet      # Canonical metadata store (updated by all phases)
```

## Troubleshooting

- Missing CUDAExecutionProvider
  - Ensure `onnxruntime-gpu` is installed and `onnxruntime` CPU is uninstalled.
- Torch reports no CUDA
  - Check `nvidia-smi` and match Torch CUDA build to your driver.
- OCR is slow or falls back to CPU
  - Confirm ORT providers include CUDAExecutionProvider and that `accel_type='CUDA'` is used.
- Out of memory
  - Lower `batch_size` for Phase‑2, reduce `GLOSSAPI_IMAGES_SCALE`, or split inputs.
