# Copilot instructions for GlossAPI

## Project overview
- GlossAPI is a multi-phase academic PDF pipeline that produces cleaned Markdown plus optional OCR/math enrichment. The public entrypoint is `Corpus` (src/glossapi/corpus/corpus_orchestrator.py) which composes phase mixins (src/glossapi/corpus/phase_*.py).
- Phase flow is: download → extract → clean → ocr/math → section → annotate → export (see `Corpus` methods in corpus_orchestrator and mixins).

## Architecture & data flow (important)
- Phase-1 extraction uses Docling via `GlossExtract` (src/glossapi/gloss_extract.py). It defaults to the “safe” PyPDFium backend when OCR/enrichment is off and switches to the Docling backend when OCR/math is requested (phase1_backend="auto").
- Cleaning is Rust-powered: `CleanPhaseMixin.clean()` loads `glossapi_rs_cleaner` and will auto-build it via `maturin develop` if missing (src/glossapi/corpus/phase_clean.py). The Rust report becomes metrics and sets `needs_ocr` in the metadata parquet.
- Phase-3 OCR/Math (`OcrMathPhaseMixin.ocr()`) uses the metadata parquet’s `needs_ocr` flag to pick OCR targets. Math enrichment consumes Docling JSON files emitted by Phase-1; DeepSeek backend skips math (src/glossapi/corpus/phase_ocr_math.py).
- Metadata parquet is discovered/bootstrapped via `ParquetSchema` and stored under output/download_results (see corpus_orchestrator + phase_clean). Keep updates consistent with that file.

## Critical workflows
- Quick smoke test: `pytest tests/test_pipeline_smoke.py` (README.md).
- Full pipeline demo: use `Corpus` in a short script, chaining `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `export()` (README.md and src/glossapi/corpus/corpus_orchestrator.py).
- Environment setup is mode-driven: `dependency_setup/setup_glossapi.sh --mode {vanilla|rapidocr|deepseek}`. DeepSeek mode expects extra env vars and weights (README.md).

## Project-specific conventions
- Lazy imports are intentional: top-level `glossapi.__init__` avoids heavy Docling imports and uses `__getattr__` to load `Corpus`/models on demand. Preserve this pattern for new modules.
- Phase methods are the stable API surface. Prefer extending phase mixins or `Corpus` helpers rather than adding ad-hoc scripts that bypass `Corpus` (see README “Corpus usage contract”).
- Skiplist handling: phases consult a skiplist file in the output directory via `_SkiplistManager` to avoid reprocessing. Ensure new phases respect it (src/glossapi/corpus/phase_*).

## Integration points & external deps
- Docling + RapidOCR are optional and can be skipped with `GLOSSAPI_SKIP_DOCLING_BOOT=1` (src/glossapi/__init__.py). Phase-1 will still work with the safe PyPDFium backend.
- GPU OCR requires torch CUDA + onnxruntime-gpu; `_gpu_preflight()` enforces availability when OCR/enrichment is requested (src/glossapi/corpus/phase_extract.py).
- Rust extensions live under rust/ and are imported from Python via `maturin develop` in `CleanPhaseMixin`.

## Examples to follow
- Extraction config + backend selection: `ExtractPhaseMixin.prime_extractor()` and `extract()` (src/glossapi/corpus/phase_extract.py).
- Rust cleaner integration and metrics merge: `CleanPhaseMixin.clean()` (src/glossapi/corpus/phase_clean.py).
- OCR selection logic from parquet and skiplist enforcement: `OcrMathPhaseMixin.ocr()` (src/glossapi/corpus/phase_ocr_math.py).
