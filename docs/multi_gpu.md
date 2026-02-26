# Multi‑GPU & Benchmarking

GlossAPI can scale across multiple visible GPUs. Faster GPUs drain more work from a shared queue of **absolute
file paths**, so no worker rescans directories.

## Extract (Phase‑1) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

- Workers are bound using `CUDA_VISIBLE_DEVICES=<id>` and run Docling on `cuda:0` relative to each worker.
- Threads auto‑tune when `num_threads=None` (roughly `min(cpu_count, 2 * #GPUs)`). Override explicitly if needed.
- The controller persists extraction progress in `download_results/download_results.parquet` after each reported
  batch, so interrupted runs can resume cleanly without ad-hoc checkpoint files.
- Worker batches requeue failed files and report `problematic` lists back to the controller, preventing silent loss.
- Periodic summaries log processed/problematic counts and queue size every ~30s for easier monitoring.

## Phase‑2 (Math) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')
c.ocr(use_gpus='multi', math_batch_size=12)
```

- Spawns math workers that bind to their GPU via `CUDA_VISIBLE_DEVICES`. Formula decoding runs on `cuda:0` relative to each worker.
- Each worker writes a marker file (`logs/math_workers/gpu<N>.current`) containing the stems it is processing and keeps an append-only log in `logs/math_workers/` (override with `GLOSSAPI_WORKER_LOG_DIR`).
- Crashed workers are respawned automatically; control the retry budget per GPU with `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`). Use `GLOSSAPI_WORKER_LOG_VERBOSE=0` to silence the banner that prints the binding info.
- When a device exceeds the respawn cap, remaining stems are added to the fatal skip-list and their artifacts are quarantined under `downloads/problematic_math/` and `json/problematic_math/` for follow-up.

## macOS / Apple Silicon (MPS)

Multi-GPU is CUDA-only. On macOS with Apple Silicon, use single-GPU mode with MPS:

```python
c.extract(input_format='pdf', accel_type='MPS', force_ocr=True)
c.ocr(device='mps')
```

For DeepSeek OCR v2 (MLX-based), the backend manages MPS natively — set `GLOSSAPI_DEEPSEEK2_DEVICE=mps` (default).

## OCR Language Selection

Set `GLOSSAPI_OCR_LANGS` (comma-separated) to control which languages RapidOCR recognizes:

```bash
export GLOSSAPI_OCR_LANGS=el,en
```

This is picked up by `extract()` and `prime_extractor()` during Phase-1.

## Provider & Device Checks

- **CUDA:** ONNXRuntime providers must include `CUDAExecutionProvider`. `torch.cuda.is_available()` must return `True`.
- **MPS (macOS):** `torch.backends.mps.is_available()` must return `True`. Use `accel_type='MPS'`.
- **CPU:** No GPU provider needed. Use `accel_type='CPU'`.

## Benchmarking Tips

- Use `benchmark_mode=True` for `extract(...)` to skip per‑doc/page metrics (reduces I/O and profiling overhead).
- Pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to avoid CPU oversubscription on multi‑GPU nodes.
- Keep caches (`HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`) on fast disks.
