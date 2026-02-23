# Quickstart

This page shows the most common tasks in a few lines each.

## Phase‑1 extraction profiles

### Stable (PyPDFium, size‑1)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA')  # OCR is off by default (Linux/Windows)
# macOS Apple Silicon (Metal/MPS): use accel_type='MPS'
```

This keeps Docling’s native parser out of the hot path and is the recommended
mode when you prioritise stability.

### Throughput (Docling, batched)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', phase1_backend='docling')  # Linux/Windows
# macOS Apple Silicon (Metal/MPS): use accel_type='MPS'
```

`phase1_backend='docling'` streams multiple PDFs through Docling’s converter and
should be used when you are comfortable trading some stability for throughput.

### Multi‑GPU

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi')  # workers share a queued file list
```

Workers report per-batch summaries and extraction progress is persisted into
`download_results/download_results.parquet`, so you can restart multi-GPU runs
without losing progress (no extra checkpoint files required).

## GPU OCR (opt-in)

> **RapidOCR / Docling path** — uses `accel_type` to select the GPU runtime:

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', force_ocr=True)  # Linux/Windows
# macOS Apple Silicon (Metal/MPS):
c.extract(input_format='pdf', accel_type='MPS', force_ocr=True)
# or multi-GPU batching:
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

> **VLM backends** (`deepseek-ocr`, `deepseek-ocr-2`, `glm-ocr`, `olmocr`, `mineru`) — GPU is selected
> automatically based on platform (MPS on macOS Apple Silicon, CUDA on Linux) or forced
> via backend-specific env vars. Use `c.ocr(backend=...)` instead of `c.extract(accel_type=...)`:

```python
c.ocr(backend='deepseek-ocr')  # macOS: MLX/MPS auto-selected; Linux/Windows: CUDA/vLLM
c.ocr(backend='glm-ocr')       # macOS: MLX/MPS auto-selected
c.ocr(backend='olmocr')        # macOS: MLX/MPS; Linux: CUDA/vLLM
c.ocr(backend='mineru')        # macOS: MPS; Linux: CUDA; CPU fallback available
```

## Phase‑2 Math Enrichment (from JSON)

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')  # same folder for input/output is fine

# Emit JSON/indices first (JSON now defaults on; request the index explicitly)
c.extract(input_format='pdf', accel_type='CUDA', emit_formula_index=True)   # Linux/Windows
# c.extract(input_format='pdf', accel_type='MPS', emit_formula_index=True)  # macOS Apple Silicon

# Enrich math/code on GPU and write enriched Markdown into markdown/<stem>.md
# Phase-2 math enrichment only applies to RapidOCR (not VLM backends).
c.formula_enrich_from_json(device='cuda', batch_size=12)   # Linux/Windows
# c.formula_enrich_from_json(device='mps', batch_size=12)  # macOS Apple Silicon
```

Progress (downloaded, OCRed, math-enriched) now lives in `download_results/download_results.parquet`; rerun `c.ocr(..., reprocess_completed=True)` whenever you need to force already successful rows back through OCR or math.

## Full Pipeline (download → extract → clean/ocr → section → annotate)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.download(url_column='url')         # optional, if you have URLs parquet
c.extract(input_format='pdf')        # Phase‑1 (no OCR by default)
c.clean()                            # compute quality; filter badness
c.ocr()                              # re‑extract bad ones and enrich math/code
c.section()                          # to parquet
c.annotate()                         # classify/annotate sections
c.jsonl_sharded(                     # export to zstd-compressed JSONL shards
    'OUT/export',
    compression='zstd',
)
```

See [OCR & Math Enrichment](ocr_and_math_enhancement.md) for GPU details, batch sizes, and artifact locations.

### Convenience shortcut

`process_all()` chains `extract → section → annotate` in one call (skips `clean()` and `ocr()`):

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.process_all(input_format='pdf', download_first=False, annotation_type='auto')
```

### JSONL Export

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')

# Single JSONL file
c.jsonl('OUT/export/output.jsonl', text_key='text')

# Sharded JSONL with metadata
c.jsonl_sharded(
    'OUT/export',
    shard_size_bytes=500 * 1024 * 1024,
    shard_prefix='train',
    compression='zstd',
    text_key='document',
    metadata_key='pipeline_metadata',
    metadata_fields=['filter', 'greek_badness_score', 'needs_ocr'],
    source_metadata_path='OUT/source_metadata.parquet',
    source_metadata_key='source_metadata',
    source_metadata_fields=['filename', 'language', 'handle_url'],
)
```

Resulting `.jsonl.zst` shards are streamable with HuggingFace Datasets:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="OUT/export/train-*.jsonl.zst", streaming=True)["train"]
```

### DeepSeek-OCR

DeepSeek-OCR can be used as an OCR backend; equations are included inline in the OCR output, so Phase‑2 math is not required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek-ocr', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To avoid stub output, set `GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR=1` and `GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0`, and ensure the CLI bits are reachable:

```bash
export GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT=/path/to/deepseek-ocr/run_pdf_ocr_vllm.py
export GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON=/path/to/deepseek-ocr-venv/bin/python
export GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR=/path/to/deepseek-ocr/DeepSeek-OCR
export GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib
python -m glossapi.ocr.deepseek_ocr.preflight  # optional: validates env without running OCR
```

### DeepSeek OCR v2 (MLX/MPS)

DeepSeek OCR v2 uses Apple's MLX runtime and Metal/MPS on macOS. Equations are included inline in the OCR output, so Phase-2 math is not required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek-ocr-2', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To avoid stub output, set `GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0`.  The runner tries in-process MLX first (fastest), then CLI subprocess, then stub.  Model weights are auto-downloaded from HuggingFace if not present locally:

```bash
export GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0
export GLOSSAPI_DEEPSEEK2_DEVICE=mps
# Optional: point to local weights to skip the auto-download
# export GLOSSAPI_DEEPSEEK2_MODEL_DIR=/path/to/DeepSeek-OCR-MLX
python -m glossapi.ocr.deepseek_ocr2.preflight  # optional: validates env without running OCR
```

### GLM-OCR (MLX/MPS)

GLM-OCR is a compact 0.5B VLM for document OCR, running on Apple Silicon via MLX. Equations are included inline in the OCR output, so Phase-2 math is not required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='glm-ocr', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To avoid stub output, set `GLOSSAPI_GLMOCR_ENABLE_STUB=0`.  The runner tries in-process MLX first (fastest), then CLI subprocess, then stub.  Model weights are auto-downloaded from HuggingFace if not present locally:

```bash
export GLOSSAPI_GLMOCR_ENABLE_STUB=0
export GLOSSAPI_GLMOCR_DEVICE=mps
# Optional: point to local weights to skip the auto-download
# export GLOSSAPI_GLMOCR_MODEL_DIR=/path/to/GLM-OCR-MLX
python -m glossapi.ocr.glm_ocr.preflight  # optional: validates env without running OCR
```

### OlmOCR-2 (vLLM / MLX)

OlmOCR-2 is a high-accuracy VLM-based OCR (Qwen2.5-VL fine-tune). It supports CUDA (vLLM) and
Apple Silicon (MLX). Equations are included inline in the OCR output, so Phase-2 math is not
required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='olmocr', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

**macOS (MLX/MPS):** set `GLOSSAPI_OLMOCR_ENABLE_STUB=0`.  The runner tries in-process MLX first
(fastest), then MLX CLI subprocess, then stub.  Model weights are auto-downloaded from HuggingFace
if not present locally:

```bash
export GLOSSAPI_OLMOCR_ENABLE_STUB=0
export GLOSSAPI_OLMOCR_DEVICE=mps
# Optional: point to local MLX weights to skip the auto-download
# export GLOSSAPI_OLMOCR_MLX_MODEL_DIR=/path/to/olmOCR-MLX
python -m glossapi.ocr.olmocr.preflight  # optional: validates env without running OCR
```

**Linux (CUDA/vLLM):** enable the CLI runner and point to model weights:

```bash
export GLOSSAPI_OLMOCR_ENABLE_STUB=0
export GLOSSAPI_OLMOCR_ENABLE_OCR=1
export GLOSSAPI_OLMOCR_MODEL_DIR=/path/to/model_weights/olmocr
python -m glossapi.ocr.olmocr.preflight  # optional: validates env without running OCR
```

### MinerU OCR

MinerU (magic-pdf) can be used as an OCR backend; equations are included inline in the OCR output, so Phase‑2 math is not required and any math flags are ignored.

On macOS, install MinerU GPU extras in a Python 3.10–3.13 venv:

```bash
pip install -U "mineru[all]"
```

Recommended: use Python 3.11 for the MinerU venv to avoid dependency gaps.

Ensure GlossAPI is installed in the active venv:

```bash
pip install -e .
```

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='mineru', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To use the CLI directly, set `GLOSSAPI_MINERU_ENABLE_OCR=1` (and optionally `GLOSSAPI_MINERU_ENABLE_STUB=0`) and ensure `magic-pdf` is on PATH or set `GLOSSAPI_MINERU_COMMAND`.

To force the GPU backend on Apple Silicon:

```bash
export GLOSSAPI_MINERU_BACKEND="hybrid-auto-engine"
export GLOSSAPI_MINERU_DEVICE_MODE="mps"
export MINERU_TOOLS_CONFIG_JSON="$GLOSSAPI_WEIGHTS_ROOT/mineru/magic-pdf.json"
python -m glossapi.ocr.mineru.preflight
```
