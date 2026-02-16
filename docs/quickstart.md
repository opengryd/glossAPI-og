# Quickstart

This page shows the most common tasks in a few lines each.

## Phase‑1 extraction profiles

### Stable (PyPDFium, size‑1)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA')  # OCR is off by default
# macOS (Metal): use accel_type='MPS'
```

This keeps Docling’s native parser out of the hot path and is the recommended
mode when you prioritise stability.

### Throughput (Docling, batched)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', phase1_backend='docling')
# macOS (Metal): use accel_type='MPS'
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

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', force_ocr=True)
# macOS (Metal): accel_type='MPS'
# or reuse multi-GPU batching
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

## Phase‑2 Math Enrichment (from JSON)

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')  # same folder for input/output is fine

# Emit JSON/indices first (JSON now defaults on; request the index explicitly)
c.extract(input_format='pdf', accel_type='CUDA', emit_formula_index=True)

# Enrich math/code on GPU and write enriched Markdown into markdown/<stem>.md
c.formula_enrich_from_json(device='cuda', batch_size=12)
# macOS (Metal): device='mps'
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

### DeepSeek OCR

DeepSeek can be used as an OCR backend; equations are included inline in the OCR output, so Phase‑2 math is not required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To avoid stub output, set `GLOSSAPI_DEEPSEEK_ALLOW_CLI=1` and `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0`, and ensure the CLI bits are reachable:

```bash
export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT=/path/to/deepseek-ocr/run_pdf_ocr_vllm.py
export GLOSSAPI_DEEPSEEK_TEST_PYTHON=/path/to/deepseek-venv/bin/python
export GLOSSAPI_DEEPSEEK_MODEL_DIR=/path/to/deepseek-ocr/DeepSeek-OCR
export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib
python -m glossapi.ocr.deepseek.preflight  # optional: validates env without running OCR
```

### DeepSeek OCR v2 (MLX/MPS)

DeepSeek OCR v2 uses Apple's MLX runtime and Metal/MPS on macOS. Equations are included inline in the OCR output, so Phase-2 math is not required and any math flags are ignored.

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek-ocr-2', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → OCR only for bad files; math is included inline in the Markdown
```

To avoid stub output, set `GLOSSAPI_DEEPSEEK2_ALLOW_STUB=0`.  The runner tries in-process MLX first (fastest), then CLI subprocess, then stub.  Model weights are auto-downloaded from HuggingFace if not present locally:

```bash
export GLOSSAPI_DEEPSEEK2_ALLOW_STUB=0
export GLOSSAPI_DEEPSEEK2_DEVICE=mps
# Optional: point to local weights to skip the auto-download
# export GLOSSAPI_DEEPSEEK2_MODEL_DIR=/path/to/DeepSeek-OCR-MLX
python -m glossapi.ocr.deepseek_ocr2.preflight  # optional: validates env without running OCR
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

To use the CLI directly, set `GLOSSAPI_MINERU_ALLOW_CLI=1` (and optionally `GLOSSAPI_MINERU_ALLOW_STUB=0`) and ensure `magic-pdf` is on PATH or set `GLOSSAPI_MINERU_COMMAND`.

To force the GPU backend on Apple Silicon:

```bash
export GLOSSAPI_MINERU_BACKEND="hybrid-auto-engine"
export GLOSSAPI_MINERU_DEVICE_MODE="mps"
export MINERU_TOOLS_CONFIG_JSON="$GLOSSAPI_WEIGHTS_ROOT/mineru/magic-pdf.json"
python -m glossapi.ocr.mineru.preflight
```
