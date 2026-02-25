# Troubleshooting

## OCR runs on CPU

- Verify ONNXRuntime GPU: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — must include `CUDAExecutionProvider` (Linux/Windows) or `CoreMLExecutionProvider` (macOS).
- Ensure CPU ORT wheel is not installed on CUDA systems: `pip uninstall -y onnxruntime`.
- For `c.extract()`: make sure you pass `accel_type='CUDA'` (or `use_gpus='multi'`) on CUDA; on macOS use `accel_type='MPS'`.
- For `c.ocr(backend='rapidocr')`: the accelerator is **auto-detected** — `mps` on macOS, `cuda` on Linux/Windows when available. No explicit `accel_type` argument is required for this code path.
- If `CoreMLExecutionProvider` is missing on macOS (e.g. ORT built without the CoreML EP), the pipeline degrades to CPU ONNX.  Consider using the native `VNRecognizeTextRequest` path instead:
  ```python
  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
  from glossapi.ocr.utils.vision_ocr import recognize_pages_parallel, is_available
  ```
  See [Apple Vision Framework](configuration.md#apple-vision-framework-macos-only) for details.

## Torch doesn’t see the GPU

- CUDA: Check `nvidia-smi` and driver installation. Match Torch CUDA build to your driver; see getting_started.md for the recommended wheel.
- macOS: Ensure you installed a PyTorch build with MPS support and that `torch.backends.mps.is_available()` returns True.

## RapidOCR font download failure

- The first OCR call might download a visualization font. Ensure egress is allowed; the file is cached afterwards.

## MinerU (magic-pdf) MFR errors with newer Transformers

- Symptom: AttributeError on `past_key_values[0][0].shape` or `cache_position` during MFR Predict.
- Cause: Newer `transformers` switched to `EncoderDecoderCache` and adds `cache_position`, while UnimerNet expects legacy tuples.
- Fix: Re-run MinerU setup with the patcher enabled (setup script applies a compatibility patch automatically in mineru mode).
- If you must pin: keep a dedicated MinerU venv and use a Transformers version before the cache API change.

## MinerU missing runtime deps

- Errors like `No module named 'matplotlib'` or `rapid_table` indicate optional runtime deps needed by MinerU backends.
- Fix: re-run MinerU setup; it installs the runtime extras and patches OCR weight mapping automatically.

## Out of memory

- Lower Phase‑2 `batch_size` (e.g., 8) and reduce inline `GLOSSAPI_FORMULA_BATCH`.
- Reduce `GLOSSAPI_IMAGES_SCALE` (e.g., 1.1–1.2).
- Split large batches or files.

## Phase‑2 math memory pressure (Apple Silicon)

- After each document is enriched, GlossAPI calls `torch.mps.synchronize()` +
  `torch.mps.empty_cache()` to compact the Metal allocator heap.  If you still see
  swap activity with large batches, lower `math_batch_size` (default `8`) or
  `GLOSSAPI_FORMULA_BATCH` (default `16`).
- The Phase-2 LRU page cache now holds up to 16 pages (up from 4).  If unified memory is
  very constrained (8 GB Mac with many other apps open), reduce it by setting
  `GLOSSAPI_ENRICH_PAGE_CACHE_SIZE` is not currently exposed — lower `batch_size` instead
  to let the GC reclaim buffers between batches.

## Worker respawn limit reached

- When a GPU crashes repeatedly, the controller stops respawning it after `GLOSSAPI_MATH_RESPAWN_CAP` attempts. Any pending stems are added to the skip‑list and their inputs are copied to `downloads/problematic_math/` (PDFs) and `json/problematic_math/` (Docling artifacts); inspect those folders, address the issue, then rerun `Corpus.ocr(..., reprocess_completed=True)` or move the quarantined files back into `downloads/`.
- Check the corresponding worker log under `logs/math_workers/` (or the directory set via `GLOSSAPI_WORKER_LOG_DIR`) for stack traces and the active stem list stored in `gpu<N>.current`.

## DeepSeek-OCR V1 (MLX/macOS) issues

- **`mlx` import error on non-Apple Silicon:** The MLX path requires Apple Silicon (M1+). On Linux/Windows use the CUDA/vLLM path instead (`pip install '.[deepseek-ocr]'`).
- **Model not found:** Set `GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR` to point to your local MLX-formatted weights directory (`deepseek-ocr-1-mlx/`), or let it auto-download from `mlx-community/DeepSeek-OCR-8bit` on HuggingFace.
- **Wrong Python:** If the MLX venv differs from the main venv, set `GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON` to the correct binary.
- **MPS device error:** Ensure `GLOSSAPI_DEEPSEEK_OCR_DEVICE=mps` (default on macOS). Check `torch.backends.mps.is_available()`.
- **Stub output (placeholder text instead of real OCR):** The stub runner is on by default. Set `GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0` to force real inference.
- **Preflight checker:** Run `python -m glossapi.ocr.deepseek_ocr.preflight` to validate your environment.

## DeepSeek OCR v2 (MLX/macOS) issues

- **`mlx` import error on non-Apple Silicon:** DeepSeek OCR v2 requires Apple Silicon (M1+) and the MLX framework. It won't work on Intel Macs or Linux.
- **Model not found:** Set `GLOSSAPI_DEEPSEEK2_MODEL_DIR` to point to your local MLX-formatted weights, or let it auto-download from HuggingFace.
- **Wrong Python:** If the MLX venv differs from the main venv, set `GLOSSAPI_DEEPSEEK2_PYTHON` to the correct binary.
- **MPS device error:** Ensure `GLOSSAPI_DEEPSEEK2_DEVICE=mps` (default). Check `torch.backends.mps.is_available()`.

## GLM-OCR (MLX/macOS) issues

- **`Model type glm_ocr not supported`:** GLM-OCR requires `mlx-vlm>=0.3.12`. Run `pip install --upgrade mlx-vlm` to update.
- **`mlx` import error on non-Apple Silicon:** GLM-OCR requires Apple Silicon (M1+) and the MLX framework. It won't work on Intel Macs or Linux.
- **Model not found:** Set `GLOSSAPI_GLMOCR_MODEL_DIR` to point to your local MLX-formatted weights, or let it auto-download from HuggingFace (`mlx-community/GLM-OCR-4bit`).
- **Wrong Python:** If the MLX venv differs from the main venv, set `GLOSSAPI_GLMOCR_PYTHON` to the correct binary.
- **MPS device error:** Ensure `GLOSSAPI_GLMOCR_DEVICE=mps` (default). Check `torch.backends.mps.is_available()`.
- **Preflight checker:** Run `python -m glossapi.ocr.glm_ocr.preflight` to validate your environment.

## OlmOCR-2 issues

- **`mlx` import error on non-Apple Silicon:** The MLX inference path requires Apple Silicon (M1+). On Linux, only the vLLM path is supported.
- **Stub output on CUDA:** The vLLM in-process and CLI strategies require `GLOSSAPI_OLMOCR_ENABLE_OCR=1` (for the OlmOCR pipeline subprocess) and `GLOSSAPI_OLMOCR_ENABLE_STUB=0`. Set both to force real inference.
- **In-process vLLM not chosen on Linux:** Ensure `vllm` is importable in the active environment. The runner skips `vllm` gracefully if it is not importable.
- **Model not found (CUDA):** Set `GLOSSAPI_OLMOCR_MODEL_DIR` to the local CUDA weights directory, or ensure `GLOSSAPI_WEIGHTS_ROOT` is set so the pipeline resolves `$GLOSSAPI_WEIGHTS_ROOT/olmocr`.
- **Model not found (MLX):** Set `GLOSSAPI_OLMOCR_MLX_MODEL_DIR` to the local MLX-formatted weights directory. Without it, weights are auto-downloaded from `mlx-community/olmOCR-2-7B-1025-4bit`.
- **`libcudart.so.12 not found`:** Set `GLOSSAPI_OLMOCR_LD_LIBRARY_PATH=/usr/local/cuda/lib64` (or wherever your CUDA runtime libraries live) to prepend it to `LD_LIBRARY_PATH` for CLI subprocesses.
- **Out-of-memory (vLLM):** Lower `GLOSSAPI_OLMOCR_GPU_MEMORY_UTILIZATION` (default `0.85`) to reduce KV-cache pre-allocation. For multi-GPU setups, vLLM's tensor parallelism is controlled directly from its own configuration.
- **Wrong Python for CLI:** If the OlmOCR venv differs from the main venv, set `GLOSSAPI_OLMOCR_PYTHON` to the correct interpreter path.
- **Preflight checker:** Run `python -m glossapi.ocr.olmocr.preflight` to validate your environment (checks olmocr package, poppler, CUDA/MLX availability, and env flags).

## MinerU backend / device issues

- **Backend selection:** Set `GLOSSAPI_MINERU_BACKEND` to override auto-detection. Set `GLOSSAPI_MINERU_DEVICE_MODE` (or `GLOSSAPI_MINERU_DEVICE`) to force `cuda`, `mps`, or `cpu`.
- **`magic-pdf` not found:** Ensure MinerU is installed and `magic-pdf` is on `PATH`, or set `GLOSSAPI_MINERU_COMMAND` to its absolute path.

## Stub runners producing empty output

- By default, DeepSeek-OCR, GLM-OCR, MinerU, and OlmOCR allow stub fallback (`*_ENABLE_STUB=1`). This means OCR may silently produce placeholder output instead of real results.
- To force real OCR, set `GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0` (or `GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0`, `GLOSSAPI_GLMOCR_ENABLE_STUB=0`, `GLOSSAPI_OLMOCR_ENABLE_STUB=0`, `GLOSSAPI_MINERU_ENABLE_STUB=0`) **and** enable the CLI runner with `*_ENABLE_OCR=1`.

## Docling import slow or crashes at startup

- `import glossapi` patches Docling at boot time. Set `GLOSSAPI_SKIP_DOCLING_BOOT=1` to skip this when you don't need Docling (e.g., running export-only or section-only phases).

## Where are my files?

- Enriched Markdown overwrites `markdown/<stem>.md` — there is never a second copy.
- Cleaned Markdown: `clean_markdown/`.
- JSON / indices / LaTeX maps: `json/`. Metrics: `json/metrics/`.
- Per-file sidecars: `sidecars/extract/`, `sidecars/triage/`, `sidecars/math/`.
- Quarantined files: `downloads/problematic_math/`, `json/problematic_math/`.
- JSONL exports: `export/`.
- Download metadata: `download_results/download_results.parquet`.
