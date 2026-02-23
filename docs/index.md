# GlossAPI Documentation

Welcome to the docs for **GlossAPI**, the [GFOSS](https://gfoss.eu/) pipeline for turning academic PDFs into clean Markdown and structured metadata artifacts. GlossAPI combines Python orchestration with Rust-powered quality metrics and supports four OCR backends for GPU-accelerated document processing.

## Start here

- [Onboarding Guide](getting_started.md) — prerequisites, install choices, and first run.
- [Quickstart Recipes](quickstart.md) — common extraction/OCR flows in copy-paste form.
- [Lightweight PDF Corpus](lightweight_corpus.md) — 20 one-page PDFs for smoke testing without Docling or GPUs.

## Learn the pipeline

- [Pipeline Overview](pipeline.md) explains each stage, the `Corpus` orchestrator, and the emitted artifacts.
- [OCR & Math Enrichment](ocr_and_math_enhancement.md) covers all six OCR backends (RapidOCR, DeepSeek-OCR, DeepSeek OCR v2, GLM-OCR, OlmOCR-2, MinerU) and Phase-2 formula enrichment.
- [Multi-GPU & Benchmarking](multi_gpu.md) shares scaling, worker management, and scheduling tips.

## Configure and debug

- [Configuration](configuration.md) lists all `GLOSSAPI_*` environment variables and tuning knobs.
- [Troubleshooting](troubleshooting.md) captures the most common pitfalls and their fixes.
- [AWS Job Distribution](aws_job_distribution.md) describes large-scale scheduling on multi-node GPU farms.

## Reference

- [Corpus API](api_corpus_tmp.md) details public methods, constructor parameters, and phase signatures.
- [Math Enrichment Runtime](math_enrichment_runtime.md) documents the early-stop and post-processing LaTeX policy.
- `docs/divio/` contains placeholder pages for the upcoming Divio restructuring — feel free to open PRs fleshing them out.
