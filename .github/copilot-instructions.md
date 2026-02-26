# GlossAPI — Copilot Instructions

> **This file is the single source of truth for AI-assisted development on this
> repository.** Every convention, constraint, and pattern documented here must be
> respected by Copilot (and any other LLM agent) when generating, modifying, or
> reviewing code in this project.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Name** | GlossAPI |
| **Owner** | [GFOSS / EELLAK](https://gfoss.eu/) |
| **Repo** | `eellak/glossAPI` |
| **License** | EUPL-1.2 |
| **Language mix** | Python (pipeline, CLI, OCR) + Rust (cleaning, quality metrics via PyO3/maturin) |
| **Python** | ≥ 3.8 (target 3.11–3.12; CI uses 3.11) |
| **Rust** | Stable, ABI3 bindings (Python ≥ 3.8) |
| **Build system** | maturin (PEP 517, `pyproject.toml`) |
| **CLI framework** | Typer + Rich; interactive prompts via [gum](https://github.com/charmbracelet/gum) |
| **Docs** | MkDocs Material (`mkdocs.yml`, `docs/`) |
| **Primary domain** | Academic document processing — PDF → Markdown → clean → OCR/math → section → annotate → export |
| **Language focus** | Greek-first (quality metrics, section classification), but multilingual-capable |

---

## 2. Repository Layout

```
glossAPI/
├── pyproject.toml                  # Build config (maturin), deps, optional extras, CLI entry
├── requirements.txt                # Pinned runtime requirements (RapidOCR stack)
├── mkdocs.yml                      # MkDocs Material site config
├── scripts/glossapi-cli.sh          # One-command setup + pipeline launcher
├── src/
│   ├── glossapi/                   # Core Python package
│   │   ├── __init__.py             # Public API, lazy imports, Docling patch
│   │   ├── _naming.py             # canonical_stem() — universal file-ID normalizer
│   │   ├── _pipeline.py           # Backward-compat re-export shim
│   │   ├── console.py             # Typer CLI entry point (glossapi pipeline | setup)
│   │   ├── pipeline_wizard.py     # Interactive pipeline wizard (gum-based)
│   │   ├── setup_wizard.py        # Interactive environment provisioning wizard
│   │   ├── parquet_schema.py      # Centralized Parquet schemas, validation, I/O
│   │   ├── text_sanitize.py       # LaTeX/formula sanitization (policy-driven)
│   │   ├── metrics.py             # Per-page OCR/parse timing + formula counting
│   │   ├── gloss_downloader.py    # Async concurrent file downloader (aiohttp)
│   │   ├── gloss_extract.py       # Docling-based document extraction
│   │   ├── gloss_section.py       # Section boundary parser → Parquet
│   │   ├── gloss_section_classifier.py  # ML section classifier (scikit-learn)
│   │   ├── corpus/                # Pipeline orchestrator + phase mixins
│   │   │   ├── corpus_orchestrator.py   # Corpus class (mixin composition)
│   │   │   ├── corpus_state.py          # Resume-checkpoint via metadata Parquet
│   │   │   ├── corpus_skiplist.py       # Persistent fatal-error skiplist
│   │   │   ├── corpus_utils.py          # _maybe_import_torch(), shared helpers
│   │   │   ├── phase_download.py        # DownloadPhaseMixin
│   │   │   ├── phase_extract.py         # ExtractPhaseMixin
│   │   │   ├── phase_clean.py           # CleanPhaseMixin
│   │   │   ├── phase_ocr_math.py        # OcrMathPhaseMixin
│   │   │   ├── phase_sections.py        # SectionPhaseMixin
│   │   │   ├── phase_annotate.py        # AnnotatePhaseMixin
│   │   │   └── phase_export.py          # ExportPhaseMixin
│   │   ├── ocr/                   # OCR backend wrappers
│   │   │   ├── rapidocr/          # Docling + RapidOCR ONNX (default backend)
│   │   │   ├── deepseek_ocr/      # DeepSeek-OCR via vLLM (CUDA) or MLX (MPS/macOS)
│   │   │   ├── deepseek_ocr2/     # DeepSeek-OCR v2 via MLX (MPS/macOS)
│   │   │   ├── glm_ocr/           # GLM-OCR via MLX (MPS/macOS)
│   │   │   ├── mineru/            # MinerU / magic-pdf CLI wrapper
│   │   │   ├── math/              # Formula/code enrichment (Phase-2)
│   │   │   └── utils/             # Shared OCR utilities (JSON I/O, triage, page)
│   │   ├── models/                # Bundled model weights (joblib, ONNX)
│   │   └── scripts/               # ocr_gpu_batch.py (multi-GPU helper)
│   └── ppocr/                     # Vendored PaddleOCR utilities
├── rust/
│   ├── glossapi_rs_cleaner/       # Rust crate: cleaning, mojibake, table analysis
│   └── glossapi_rs_noise/         # Rust crate: noise-based quality metrics
├── tests/                         # pytest suite
├── docs/                          # MkDocs source
├── samples/
│   ├── lightweight_pdf_corpus/    # Synthetic PDFs + expected outputs (CI golden)
│   └── eellak/                    # EELLAK ground-truth test data
├── dependency_setup/              # Environment provisioning scripts per profile
├── artifacts/                     # Pipeline output directories (gitignored runs)
└── Greek_variety_classification/  # Ancillary: Greek text classifier notebooks
```

---

## 3. Architecture & Core Design Patterns

### 3.1 Mixin-Based Corpus Orchestrator

`Corpus` is the **single stable entrypoint** for the pipeline. It composes seven
phase mixins via cooperative multiple inheritance:

```python
class Corpus(
    DownloadPhaseMixin,    # download()
    ExtractPhaseMixin,     # extract(), prime_extractor()
    CleanPhaseMixin,       # clean()
    OcrMathPhaseMixin,     # ocr()
    SectionPhaseMixin,     # section()
    AnnotatePhaseMixin,    # annotate()
    ExportPhaseMixin,      # jsonl(), jsonl_sharded()
):
    ...
```

**Rules:**
- New pipeline functionality **must** be added as a mixin in a new or existing
  `phase_*.py` file under `src/glossapi/corpus/`, then wired into the `Corpus`
  MRO in `corpus_orchestrator.py`.
- Never monkeypatch the orchestrator or bypass the phase methods. Callers keep
  the `download() → extract() → clean() → ocr() → section() → annotate() →
  export()` chain intact.
- `process_all()` chains `extract → section → annotate` (no clean/OCR). It is a
  convenience shortcut, not a replacement for explicit phase calls.

### 3.2 Canonical Stem Identity

`canonical_stem()` in `src/glossapi/_naming.py` is the **universal file-ID
normalizer**. It strips compound suffixes (`.docling.json.zst`, `.latex_map.jsonl`,
`.per_page.metrics.json`, etc.) to produce a bare document stem.

**Every module** that matches files across phases must resolve through
`canonical_stem()`. Never roll ad-hoc suffix stripping.

### 3.3 Parquet as State

- `download_results/download_results.parquet` is the canonical metadata store.
- `ParquetSchema` (`parquet_schema.py`) defines `COMMON_SCHEMA`,
  `METADATA_SCHEMA`, `DOWNLOAD_SCHEMA`, `SECTION_SCHEMA`, `CLASSIFIED_SCHEMA`.
- `ParquetSchema.ensure_metadata_parquet()` can **synthesize** metadata from
  on-disk artifacts (downloads, markdown, JSON metrics, LaTeX maps, triage/math
  sidecars) when none exists.
- Parquet writes use **atomic rename** (`tmp` + `os.replace`) and **file-based
  locks** (`filelock.FileLock` with fallback to `os.O_CREAT | os.O_EXCL`).
- Boolean columns use pandas `boolean` dtype; `_coerce_bool_value()` normalizes
  truthy/falsy representations (`"false"`, `"TRUE"`, `pd.NA`).
- `math_enriched` and `enriched_math` are treated as aliases.

### 3.4 Lazy / Optional Imports

Heavy dependencies (Docling, torch, onnxruntime, transformers, detectron2,
pyarrow) are **never imported at module level**. Use:
- `try/except ImportError` guards
- `_maybe_import_torch(*, force=False)` from `corpus_utils.py`
- Lazy `__getattr__` in `__init__.py` and `ocr/__init__.py`

This keeps `import glossapi` lightweight and allows the vanilla profile (no GPU
extras) to work.

### 3.5 Skiplist Management

`_SkiplistManager` (`corpus_skiplist.py`) maintains a persistent on-disk list of
stems that caused **fatal errors**. These are never retried.
- File: `<output_dir>/skiplists/fatal_skip.txt` (overridable via
  `GLOSSAPI_SKIPLIST_PATH`).
- Writes are atomic (`tmp` + `os.replace`).

### 3.6 Resumability

`_ProcessingStateManager` (`corpus_state.py`) reads `extract_status` and
`processing_stage` columns from the metadata Parquet to determine which files
have already been processed. Markdown files on disk serve as a fallback signal.
Phases respect skiplists and state so the pipeline can be interrupted and resumed
at any point.

### 3.7 Multi-GPU Processing

`gpu_extract_worker_queue()` (in `corpus_orchestrator.py`) supports multi-worker
extraction. Each worker:
- Binds to a single GPU via `CUDA_VISIBLE_DEVICES`
- Creates its own `Corpus` instance
- Processes batches from a shared `multiprocessing.Queue`
- Reports results via a result queue
- Auto-sets `OMP_NUM_THREADS=1` to prevent thread explosion

### 3.8 Single Canonical Markdown Location

Enriched Markdown **overwrites** plain Markdown at `markdown/<stem>.md`. There is
never a second copy. JSON and formula indexes live under `json/`.

---

## 4. Rust Crates

### 4.1 `glossapi_rs_cleaner` (`rust/glossapi_rs_cleaner/`)

| Field | Value |
|---|---|
| **Purpose** | Parallel Markdown cleaning, mojibake detection, table analysis, script classification |
| **Key deps** | `regex`, `aho-corasick`, `rayon`, `arrow`/`parquet`, `htmlentity`, `memchr` |
| **Bindings** | PyO3 ABI3 (Python ≥ 3.8) |
| **Entry** | `run_complete_pipeline()` — clean → detect tables → remove tables → write cleaned files → analyze → Parquet report |
| **Modules** | `cleaning_module`, `directory_processor`, `pipeline_module`, `table_analysis_module`, `table_remover_module` |
| **Patterns** | Aho-Corasick for fast artefact line detection; character script classification (Greek, Latin, French, Spanish, Coptic, Cyrillic); LaTeX environment stripping; Arrow RecordBatch output |

### 4.2 `glossapi_rs_noise` (`rust/glossapi_rs_noise/`)

| Field | Value |
|---|---|
| **Purpose** | Greek-specific noise-based quality metrics |
| **Key deps** | `rayon`, `walkdir`, `memmap2`, `regex` |
| **Bindings** | PyO3 ABI3 (Python ≥ 3.8) |
| **Functions** | `score_markdown_file(path)`, `score_markdown_directory(dir, n_threads)`, `score_markdown_file_detailed(path)` (24-element tuple), `score_markdown_directory_detailed(dir, n_threads)` |
| **Algorithm** | Greek vowel/consonant penalty rates, bad doubles, misplaced sigma (2.5× weight), invalid bigrams, long-word scoring, short-word excess; per-1000 normalization on Greek base codepoints; memmap I/O + rayon parallelism |

**Build convention:** Both crates are built via `maturin develop --release` or
`pip install -e .`. The `clean()` phase auto-builds `glossapi_rs_cleaner` if the
module is missing. Always use `pip install -e .` (not bare `maturin develop`) for
the Rust extensions to ensure they integrate properly with the Python package.

---

## 5. OCR Backends

| Backend | Module path | GPU | Math handling | When to use |
|---|---|---|---|---|
| **RapidOCR** (default) | `ocr.rapidocr` | CUDA / MPS / CPU | Separate Phase-2 enrichment from Docling JSON | Default; Docling + RapidOCR ONNX |
| **DeepSeek-OCR** | `ocr.deepseek_ocr` | CUDA & MPS | Inline (no Phase-2) | CUDA: vLLM; MPS: MLX in-process/CLI |
| **DeepSeek v2** | `ocr.deepseek_ocr2` | MPS (MLX) | Inline (no Phase-2) | macOS Apple Silicon |
| **GLM-OCR** | `ocr.glm_ocr` | MPS (MLX) | Inline (no Phase-2) | Lightweight 0.5B VLM OCR on macOS Apple Silicon |
| **MinerU** | `ocr.mineru` | CUDA / MPS / CPU | Inline (no Phase-2) | External `magic-pdf` CLI |
| **OlmOCR-2** | `ocr.olmocr` | CUDA (vLLM) / MPS (MLX) | Inline (no Phase-2) | High-accuracy VLM-based OCR with CUDA or Apple Silicon GPU |

**Critical policies:**
- Never OCR and math-enrich the same file in the same pass.
- DeepSeek-OCR/GLM-OCR/MinerU/OlmOCR backends inline equations — Phase-2 math enrichment is a no-op.
- RapidOCR dispatches through `Corpus.extract()` with `force_ocr=True`,
  `phase1_backend="docling"`.
- Stub runners are **allowed by default** (`*_ENABLE_STUB=1`). To force real OCR,
  set `*_ENABLE_STUB=0` and `*_ENABLE_OCR=1`.
- `ocr/__init__.py` uses lazy `__getattr__` — `import glossapi.ocr` must stay
  lightweight.

---

## 6. Pipeline Phases & Artifact Layout

### Phase Flow

```
download() → extract() → clean() → ocr() → section() → annotate() → jsonl() / jsonl_sharded()
```

### Artifact Directory

```
output_dir/
├── downloads/               # Raw downloaded files
├── download_results/        # Download metadata parquet(s)
├── markdown/                # Phase-1 extracted Markdown (overwritten by enrichment)
├── json/                    # Docling JSON + formula indexes + metrics/
├── clean_markdown/          # Phase-2 Rust-cleaned Markdown
├── sidecars/                # Per-file metadata (extract/, triage/, math/ subdirs)
├── sections/                # sections_for_annotation.parquet
├── logs/                    # Per-run log files
├── skiplists/               # fatal_skip.txt + phase-specific skiplists
├── export/                  # JSONL / sharded JSONL output (.jsonl.zst)
├── classified_sections.parquet
└── fully_annotated_sections.parquet
```

---

## 7. Environment Variables

GlossAPI is heavily configured via `GLOSSAPI_*` environment variables. Key
categories:

### GPU & Device

| Variable | Purpose |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `GLOSSAPI_DOCLING_DEVICE` | Docling device override (`cuda`, `mps`, `cpu`) |
| `GLOSSAPI_GPU_BATCH_SIZE` | Batch size for GPU extraction workers |

### Model Weights

| Variable | Purpose |
|---|---|
| `GLOSSAPI_WEIGHTS_ROOT` | Root directory for all model weights (default: `<repo>/model_weights/`). Per-backend subdirectories are derived automatically (`deepseek-ocr/`, `deepseek-ocr-mlx/`, `glm-ocr-mlx/`, `olmocr/`, `olmocr-mlx/`, `mineru/`). |

### OCR Control

| Variable | Purpose |
|---|---|
| `GLOSSAPI_IMAGES_SCALE` | Image scale for OCR |
| `GLOSSAPI_FORMULA_BATCH` | Formula enrichment batch size |
| `GLOSSAPI_MATH_RESPAWN_CAP` | Max math worker respawns (default 5) |
| `GLOSSAPI_SKIP_DOCLING_BOOT` | Skip Docling patching at import (`1` to skip) |
| `GLOSSAPI_OCR_LANGS` | Comma-separated OCR languages, e.g. `el,en` |

### VLM Backend Performance Tuning

| Variable | Purpose |
|---|---|
| `GLOSSAPI_VLM_MAX_TOKENS` | Global max-tokens cap applied to all VLM backends; per-backend vars take precedence when set. Default varies by backend (2048–4096). |
| `GLOSSAPI_VLM_RENDER_PREFETCH` | Number of pages to pre-render ahead via thread pool (1–4, default `2`). Higher values reduce GPU idle time on multi-page PDFs. |

### DeepSeek-OCR (MPS/MLX — Apple Silicon)

| Variable | Purpose |
|---|---|
| `GLOSSAPI_DEEPSEEK_OCR_DEVICE` | Force device: `cuda` / `mps` / `cpu` (auto-detected: `mps` on macOS, `cuda` elsewhere) |
| `GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL` | HuggingFace MLX model identifier (default `mlx-community/DeepSeek-OCR-8bit`) |
| `GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR` | Optional override for MLX model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-1-mlx`) |
| `GLOSSAPI_DEEPSEEK_OCR_MLX_SCRIPT` | Path to MLX inference script for subprocess execution (default: package-embedded `mlx_cli.py`) |
| `GLOSSAPI_DEEPSEEK_OCR_MAX_TOKENS` | Max tokens for DeepSeek-OCR MLX generation (default `4096`; overrides `GLOSSAPI_VLM_MAX_TOKENS`) |

### DeepSeek-OCR (CUDA/vLLM)

| Variable | Purpose |
|---|---|
| `GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR` | Enable real CLI runner |
| `GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB` | Enable stub fallback (default `1`; set `0` to disable) |
| `GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT` | Path to `run_pdf_ocr_vllm.py` |
| `GLOSSAPI_DEEPSEEK_OCR_MODEL_DIR` | Optional override for model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr`) |
| `GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON` | Python binary for DeepSeek-OCR venv |
| `GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH` | Library path for libjpeg-turbo etc. |
| `GLOSSAPI_DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION` | VRAM fraction (0.5–0.9) |
| `GLOSSAPI_DEEPSEEK_OCR_NO_FP8_KV` | Disable FP8 KV cache (`1`) |

### DeepSeek OCR v2 (MLX/macOS)

| Variable | Purpose |
|---|---|
| `GLOSSAPI_DEEPSEEK2_ENABLE_OCR` | Enable real MLX CLI runner |
| `GLOSSAPI_DEEPSEEK2_ENABLE_STUB` | Enable stub fallback (default `1`; set `0` to disable) |
| `GLOSSAPI_DEEPSEEK2_MLX_SCRIPT` | Path to MLX inference script |
| `GLOSSAPI_DEEPSEEK2_MODEL_DIR` | Optional override for MLX model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-mlx`) |
| `GLOSSAPI_DEEPSEEK2_PYTHON` | Python binary for DeepSeek v2 venv |
| `GLOSSAPI_DEEPSEEK2_DEVICE` | Device override (`mps` default) |
| `GLOSSAPI_DEEPSEEK2_MAX_TOKENS` | Max tokens for DeepSeek v2 MLX generation (default `4096`; overrides `GLOSSAPI_VLM_MAX_TOKENS`) |

### MinerU

| Variable | Purpose |
|---|---|
| `GLOSSAPI_MINERU_ENABLE_OCR` | Enable real CLI runner |
| `GLOSSAPI_MINERU_ENABLE_STUB` | Enable stub fallback (default `1`; set `0` to disable) |
| `GLOSSAPI_MINERU_COMMAND` | Override `magic-pdf` path |
| `GLOSSAPI_MINERU_MODE` | `auto` / `fast` / `accurate` |
| `GLOSSAPI_MINERU_BACKEND` | Override MinerU internal backend selection |
| `GLOSSAPI_MINERU_DEVICE_MODE` | Force device: `cuda` / `mps` / `cpu` (alias: `GLOSSAPI_MINERU_DEVICE`) |
| `GLOSSAPI_MINERU_FORMULA_ENABLE` | Set to `0` to skip formula recognition (MFR) entirely; `1` to force-enable. Injected as `formula-config.enable`. |
| `GLOSSAPI_MINERU_TABLE_ENABLE` | Set to `0` to skip table extraction; `1` to force-enable. Injected as `table-config.enable`. |
| `VIRTUAL_VRAM_SIZE` | MinerU env var (no `GLOSSAPI_` prefix) overriding detected GPU memory for `batch_ratio` scaling. Auto-injected by GlossAPI on MPS based on physical RAM after the `setup_glossapi.sh` MPS branch patch (`8–15 GiB → "6"`, `16–23 GiB → "8"`, `≥․24 GiB → "12"`). |

### GLM-OCR

| Variable | Purpose |
|---|---|
| `GLOSSAPI_GLMOCR_ENABLE_OCR` | Enable real GLM-OCR MLX CLI subprocess |
| `GLOSSAPI_GLMOCR_ENABLE_STUB` | Enable stub fallback (default `1`; set `0` to disable) |
| `GLOSSAPI_GLMOCR_PYTHON` | Python binary for GLM-OCR venv |
| `GLOSSAPI_GLMOCR_MODEL_DIR` | Optional override for model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/glm-ocr-mlx`) |
| `GLOSSAPI_GLMOCR_MLX_MODEL` | HuggingFace MLX model identifier (default `mlx-community/GLM-OCR-4bit`) |
| `GLOSSAPI_GLMOCR_MLX_SCRIPT` | Path to MLX inference script for subprocess execution |
| `GLOSSAPI_GLMOCR_DEVICE` | Device override (`mps`, `cpu`) |
| `GLOSSAPI_GLMOCR_MAX_TOKENS` | Max tokens for GLM-OCR generation (default `2048`; overrides `GLOSSAPI_VLM_MAX_TOKENS`) |

### OlmOCR-2

| Variable | Purpose |
|---|---|
| `GLOSSAPI_OLMOCR_ENABLE_OCR` | Enable real OlmOCR pipeline (external `olmocr` package) |
| `GLOSSAPI_OLMOCR_ENABLE_STUB` | Enable stub fallback (default `1`; set `0` to disable) |
| `GLOSSAPI_OLMOCR_PYTHON` | Python binary for OlmOCR venv |
| `GLOSSAPI_OLMOCR_MODEL` | HuggingFace model identifier (default `allenai/olmOCR-2-7B-1025-FP8`) |
| `GLOSSAPI_OLMOCR_MODEL_DIR` | Optional override for CUDA model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/olmocr`) |
| `GLOSSAPI_OLMOCR_SERVER` | URL of external vLLM server |
| `GLOSSAPI_OLMOCR_API_KEY` | API key for external vLLM server |
| `GLOSSAPI_OLMOCR_GPU_MEMORY_UTILIZATION` | VRAM fraction for vLLM KV-cache (default `0.85`) |
| `GLOSSAPI_OLMOCR_TARGET_IMAGE_DIM` | Longest-side dimension for PDF page rendering (OlmOCR CLI only) |
| `GLOSSAPI_OLMOCR_WORKERS` | Number of OlmOCR pipeline workers (OlmOCR CLI only) |
| `GLOSSAPI_OLMOCR_PAGES_PER_GROUP` | PDF pages per work item group (OlmOCR CLI only) |
| `GLOSSAPI_OLMOCR_VLLM_SCRIPT` | Path to vLLM CLI inference script for subprocess execution |
| `GLOSSAPI_OLMOCR_LD_LIBRARY_PATH` | Extra library paths prepended to `LD_LIBRARY_PATH` for CLI subprocesses (e.g. `/usr/local/cuda/lib64`) |
| `GLOSSAPI_OLMOCR_MLX_MODEL` | HuggingFace MLX model identifier (default `mlx-community/olmOCR-2-7B-1025-4bit`) |
| `GLOSSAPI_OLMOCR_MLX_MODEL_DIR` | Optional override for MLX model weights directory (default: `$GLOSSAPI_WEIGHTS_ROOT/olmocr-mlx`) |
| `GLOSSAPI_OLMOCR_MLX_SCRIPT` | Path to MLX inference script for subprocess execution |
| `GLOSSAPI_OLMOCR_DEVICE` | Device override (`cuda`, `mps`, `cpu`) |
| `GLOSSAPI_OLMOCR_MAX_TOKENS` | Max tokens for OlmOCR vLLM generation (default `2048`; overrides `GLOSSAPI_VLM_MAX_TOKENS`) |
| `GLOSSAPI_OLMOCR_MLX_MAX_TOKENS` | Max tokens for OlmOCR MLX (Apple Silicon) generation (default `2048`; overrides `GLOSSAPI_VLM_MAX_TOKENS`) |

### LaTeX Policy — Early Stop (during decoding)

| Variable | Purpose |
|---|---|
| `GLOSSAPI_LATEX_EARLYSTOP` | Enable early stop (default `1` = enabled) |
| `GLOSSAPI_LATEX_MAX_REPEAT` | Stop if last token repeats more than N times (default 50) |
| `GLOSSAPI_LATEX_MAX_CHARS` | Char cap during generation (default 3000) |
| `GLOSSAPI_LATEX_LEN_STRIDE` | Check frequency — every N steps (default 16) |
| `GLOSSAPI_LATEX_MAX_NEW_TOKENS` | Hard token cap for the decoder (optional) |

### LaTeX Policy — Post-Processing (after decoding)

| Variable | Purpose |
|---|---|
| `GLOSSAPI_LATEX_POST_ONLY_FAILED` | Apply post-processing only if gating triggers (default `1`) |
| `GLOSSAPI_LATEX_POST_REPEAT_GATE` | Treat as failed if tail run > gate (default 50) |
| `GLOSSAPI_LATEX_POST_WINDDOWN` | Wind down tail run to this count (default 12) |
| `GLOSSAPI_LATEX_POST_MAX_CHARS` | Treat as failed if len > cap, trim to cap (default 3000) |

### Misc

| Variable | Purpose |
|---|---|
| `GLOSSAPI_SKIPLIST_PATH` | Override skiplist file path |
| `GLOSSAPI_WORKER_LOG_*` | Worker logging configuration |
| `GLOSSAPI_TRIAGE_HEURISTIC` | Re-enable legacy triage heuristic (`1`) |
| `GLOSSAPI_SETUP_SIMPLE` | Use simple TTY prompts instead of gum |

---

## 8. CLI

```bash
glossapi              # Launches pipeline wizard (default)
glossapi pipeline     # Interactive phase-selection wizard
glossapi setup        # Environment provisioning wizard
```

- Entry point: `glossapi = "glossapi.console:app"` in `pyproject.toml`.
- `console.py` registers two Typer sub-apps (`pipeline_wizard.app`,
  `setup_wizard.app`).
- Both wizards use **gum** for rich prompts, falling back to simple TTY if gum is
  unavailable.
- Pipeline wizard presets: "Lightweight PDF smoke test", "MinerU demo", "Custom".
- Setup wizard modes: `vanilla`, `rapidocr`, `mineru`, `deepseek-ocr`,
  `deepseek-ocr-2`, `glm-ocr`.

---

## 9. Dependency Profiles

| Profile | Extras | GPU | Notes |
|---|---|---|---|
| **vanilla** | (none) | No | Minimal deps, PyPDFium backend only |
| **rapidocr** | `[rapidocr]` | CUDA / MPS / CPU | Docling + RapidOCR + ORT-GPU |
| **cuda** | `[cuda]` | CUDA | Torch + torchvision for GPU layout |
| **deepseek-ocr** | `[deepseek-ocr]` | CUDA | vLLM + transformers + accelerate |
| **deepseek-ocr-2** | (manual) | MPS (MLX) | MLX-formatted weights, macOS only |
| **glm-ocr** | (manual) | MPS (MLX) | Lightweight 0.5B VLM, macOS only |
| **mineru** | (manual) | CUDA / MPS / CPU | External `magic-pdf` CLI |
| **docs** | `[docs]` | No | MkDocs Material for doc builds |

Provisioning: `./dependency_setup/setup_glossapi.sh --mode <profile> [--venv <path>] [--run-tests]`

---

## 10. Testing Conventions

### Framework & Execution

- **Framework:** pytest
- **Quick smoke test:** `pytest tests/test_pipeline_smoke.py`
- **Lightweight (no GPU):** `pytest tests/test_corpus_phases_lightweight.py`
- **Full suite:** `pytest tests/`

### Markers

| Marker | Meaning |
|---|---|
| `@pytest.mark.rapidocr` | Requires RapidOCR/Docling execution stack |
| `@pytest.mark.deepseek_ocr` | Exercises the DeepSeek-OCR pipeline |
| (unmarked) | Vanilla — no GPU required |

### Conventions

- **Filesystem isolation:** Always use `tmp_path` fixture. Never write to real
  `artifacts/` from tests.
- **Import guards:** `pytest.importorskip("docling")`,
  `pytest.importorskip("glossapi_rs_cleaner")`,
  `pytest.importorskip("onnxruntime")` to skip tests when optional deps are
  absent.
- **Mocking heavy deps:** Use `monkeypatch` (setattr, setenv, delenv) to stub
  torch, onnxruntime, Docling, DeepSeek vLLM. Create stub classes like
  `DummyExtractor`, `FakeCorpus`.
- **Parquet roundtrip pattern:** Write seed Parquet → run pipeline step → read
  Parquet → assert column values.
- **Golden output comparison:** `samples/lightweight_pdf_corpus/expected_outputs.json`
  is the CI oracle. Update it together with `generate_pdfs.py` changes.
- **Concurrency tests:** `ThreadPoolExecutor` for testing thread-safe Parquet I/O.

### Adding New Tests

1. Place test files in `tests/test_<feature>.py`.
2. Use appropriate markers if the test requires GPU/optional deps.
3. Follow the existing seed-Parquet → run → assert pattern.
4. Register any new markers in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## 11. Coding Standards & Conventions

### Python

- **Style:** PEP 8. Use type hints for function signatures.
- **Imports:** Group as stdlib → third-party → local. Lazy-import heavy deps.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for constants and env var names.
- **Docstrings:** Use docstrings on all public classes and methods. Phase mixin
  methods should document their parameters and side effects.
- **Error handling:** Isolate per-file/per-document exceptions. Fatal errors →
  skiplist. Transient errors → retry or mark "problematic". Never let one
  document failure crash the pipeline.
- **Logging:** Use Python `logging` module. Each phase logs to its own file under
  `logs/`. Workers use `GLOSSAPI_WORKER_LOG_*` env vars for configuration.
- **No global state mutation:** Avoid module-level mutable state. Configuration
  flows through `Corpus.__init__` parameters and env vars.

### Rust

- **Build:** `maturin` + PyO3 with `abi3` bindings (min Python 3.8).
- **Parallelism:** `rayon` for directory/file-level parallelism.
- **I/O:** `memmap2` for memory-mapped file reads; `arrow`/`parquet` for
  structured output.
- **Pattern matching:** `aho-corasick` for multi-pattern line detection; `regex`
  for complex patterns.
- **Error handling:** Return `PyResult<T>` from `#[pyfunction]` entries. Panic
  only on programmer errors, never on user data.

### General

- **Atomic writes:** Always use `tmp` + `os.replace()` for file writes that must
  be crash-safe (Parquet, skiplists).
- **File locking:** Use `filelock.FileLock` (with `os.O_CREAT | os.O_EXCL`
  fallback) for concurrent Parquet access.
- **Thread clamping:** Set `OMP_NUM_THREADS=1` per GPU worker to prevent thread
  explosion.
- **No bespoke monkeypatches:** All functionality flows through the phase mixin
  API. The only sanctioned monkey-patch is `SafeRapidOcrModel` injection into
  Docling at boot time.

---

## 12. Contributing Checklist

1. **Run `pytest tests/test_pipeline_smoke.py`** for a fast end-to-end check.
2. **Run `pytest tests/test_corpus_phases_lightweight.py`** for no-GPU validation.
3. **Regenerate lightweight corpus** via `python samples/lightweight_pdf_corpus/generate_pdfs.py`
   and commit updated PDFs + manifest together.
4. **New phases** → add as mixin in `phase_<name>.py`, wire into `Corpus` MRO.
5. **New env vars** → document in `docs/configuration.md` and in this file (§7).
6. **New pytest markers** → register in `pyproject.toml`.
7. **Respect contracts:** skiplist, `canonical_stem()`, Parquet schema, single
   canonical Markdown location.
8. **Prefer `pip install -e .`** for editable installs so Rust extensions rebuild.
9. **Never commit to `master` directly.** Use feature branches and PRs.

---

## 13. Key Files Quick Reference

| What you need | Where to look |
|---|---|
| Add a pipeline phase | `src/glossapi/corpus/phase_*.py` + `corpus_orchestrator.py` |
| Modify OCR dispatch | `src/glossapi/corpus/phase_ocr_math.py` + `src/glossapi/ocr/<backend>/` |
| Change Parquet schemas | `src/glossapi/parquet_schema.py` |
| File-ID normalization | `src/glossapi/_naming.py` (`canonical_stem()`) |
| LaTeX sanitization logic | `src/glossapi/text_sanitize.py` |
| Quality scoring algorithm | `rust/glossapi_rs_noise/src/lib.rs` |
| Markdown cleaning logic | `rust/glossapi_rs_cleaner/src/` |
| CLI entry point | `src/glossapi/console.py` → `pipeline_wizard.py` / `setup_wizard.py` |
| Environment provisioning | `dependency_setup/setup_glossapi.sh` |
| CI smoke tests | `tests/test_pipeline_smoke.py`, `tests/test_corpus_phases_lightweight.py` |
| Expected test outputs | `samples/lightweight_pdf_corpus/expected_outputs.json` |
| All env var docs | `docs/configuration.md` |
| Troubleshooting | `docs/troubleshooting.md` |

---

## 14. Common Gotchas

1. **CPU-only ORT wheel conflict:** On CUDA systems, `onnxruntime` (CPU) must be
   **uninstalled** before installing `onnxruntime-gpu`. They conflict.
2. **Rust extensions with `maturin develop`:** Use `pip install -e .` instead of
   bare `maturin develop` — the latter doesn't integrate properly with the Python
   package in all cases.
3. **`_parquet_lock` parent dirs:** The lock helper must create parent directories
   before acquiring file locks. If you modify lock paths, ensure `mkdir -p`
   semantics.
4. **Legacy env vars:** `GLOSSAPI_BATCH_POLICY` / `GLOSSAPI_BATCH_MAX` still
   parse but emit deprecation warnings. Remove usage in new code.
5. **Docling patching:** `_attempt_patch_docling()` runs at import time. Skip
   with `GLOSSAPI_SKIP_DOCLING_BOOT=1` in environments without Docling.
6. **Enriched Markdown location:** Always at `markdown/<stem>.md` — phase-2
   enrichment **overwrites** the phase-1 output. Never create a second copy.
7. **Stub runners default to ON:** `*_ENABLE_STUB=1` by default for DeepSeek and
   MinerU. Tests may silently use stubs unless you explicitly set
   `*_ENABLE_STUB=0`.
8. **FlashInfer JIT on DeepSeek:** Needs `nvcc` on `PATH` and `CUDA_HOME` set.
   If problematic, disable with `VLLM_USE_FLASHINFER=0`.
9. **macOS ORT:** Use `onnxruntime==1.18.1` (no `-gpu` suffix) on macOS.
10. **Thread clamping:** Always set `OMP_NUM_THREADS=1` in multi-GPU worker
    processes to avoid thread explosion.
