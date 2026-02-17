# GlossAPI

GlossAPI is a GPU-ready document processing pipeline from [GFOSS](https://gfoss.eu/) that turns academic PDFs (and other document formats) into structured Markdown, cleans noisy text with Rust extensions, and optionally enriches math/code content via multiple OCR backends.

## Why GlossAPI

- **End-to-end pipeline** — download → extract → clean → OCR/math → section → annotate → export, all through one `Corpus` object.
- **Multi-format extraction** — PDF, DOCX, HTML, XML/JATS, PowerPoint, CSV, and Markdown via Docling or the lightweight PyPDFium backend.
- **Six OCR backends** — Docling + RapidOCR (default), DeepSeek-OCR (vLLM), DeepSeek-OCR v2 (MLX/MPS), GLM-OCR (MLX/MPS), OlmOCR-2 (vLLM/MLX), and MinerU (magic-pdf). Pick the one that fits your GPU and accuracy needs.
- **Rust-powered quality metrics** — Two Rust crates (`glossapi_rs_cleaner` for mojibake/noise cleaning, `glossapi_rs_noise` for quality scoring) keep Markdown quality predictable and fast.
- **Greek-first design** — Metadata handling and section classification are tuned for academic Greek corpora, but the pipeline works for any language.
- **Resumable & modular** — Phase methods respect skiplists and metadata parquet state, so you can resume from any stage or cherry-pick phases in custom scripts.
- **Multi-GPU support** — Scale extraction and OCR across multiple GPUs with built-in queue-based dispatching.
- **macOS Metal/MPS** — GPU acceleration on Apple Silicon via CoreML (RapidOCR/Docling), MinerU, and MPS-aware torch.
- **Sharded JSONL export** — Produce zstd-compressed shards ready for HuggingFace Datasets streaming.

## Quickstart (local repo)

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the lightweight PDF corpus (no GPU/Docling required)
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")  # Safe PyPDFium backend by default
PY
```

- Compare the generated Markdown in `artifacts/lightweight_pdf_run/markdown/`
  with `samples/lightweight_pdf_corpus/expected_outputs.json` for a fast smoke check.
- Rebuild the corpus anytime with `python samples/lightweight_pdf_corpus/generate_pdfs.py`.

### Full pipeline example

```python
from pathlib import Path
from glossapi import Corpus

corpus = Corpus(
    input_dir=Path("data/pdfs"),
    output_dir=Path("artifacts/my_run"),
)

corpus.extract(input_format="pdf")           # Phase 1 — PDF → Markdown
corpus.clean()                               # Phase 2 — Rust cleaner + metrics
corpus.ocr(backend="rapidocr")              # Phase 3 — Re-OCR bad docs + math enrichment
corpus.section()                             # Phase 4 — Extract sections → Parquet
corpus.annotate()                            # Phase 5 — ML classification
corpus.jsonl_sharded(                        # Phase 6 — Export
    output_dir=Path("artifacts/my_run/export"),
    compression="zstd",
)
```

### Corpus usage contract

`Corpus` is the organizing surface: keep contributions wired through the phase methods (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `jsonl()`, `jsonl_sharded()`). The intended use is a short script chaining those calls; avoid bespoke monkeypatches or side channels so resumability and artifact layout stay consistent.

## Pipeline Phases

| Phase | Method | Description |
| --- | --- | --- |
| Download | `corpus.download()` | Fetch PDFs from a URL parquet (resume-aware, parallel scheduler grouping). |
| Extract | `corpus.extract()` | Convert documents to Markdown. Backends: `"safe"` (PyPDFium), `"docling"`, or `"auto"`. Supports PDF, DOCX, HTML, XML/JATS, PPTX, CSV, MD. |
| Clean | `corpus.clean()` | Rust-powered cleaning and mojibake detection. Sets `needs_ocr` flag in metadata. |
| OCR / Math | `corpus.ocr()` | Re-OCR bad documents and/or enrich math. Backends: `"rapidocr"`, `"deepseek"`, `"deepseek-ocr-2"`, `"glm-ocr"`, `"olmocr"`, `"mineru"`. |
| Section | `corpus.section()` | Extract sections from Markdown into a structured Parquet. |
| Annotate | `corpus.annotate()` | Classify sections with a pre-trained model. Modes: `"text"`, `"chapter"`, `"auto"`. |
| Export | `corpus.jsonl()` / `corpus.jsonl_sharded()` | Produce JSONL (optionally zstd-compressed shards) with merged metadata. |

A convenience method `corpus.process_all()` chains extract → section → annotate in one call (optionally download first via `download_first=True`). It does not include the `clean()` or `ocr()` steps.

## OCR Backends

| Backend | Flag | GPU | Math handling | Notes |
| --- | --- | --- | --- | --- |
| **RapidOCR** | `backend="rapidocr"` | CUDA / MPS / CPU | Separate math enrichment from Docling JSON | Default. Docling + RapidOCR ONNX stack. |
| **DeepSeek** | `backend="deepseek"` | CUDA (vLLM) | Inline (equations embedded in OCR output) | Requires DeepSeek-OCR weights + vLLM. |
| **DeepSeek v2** | `backend="deepseek-ocr-2"` | MPS (MLX) | Inline (equations embedded in OCR output) | Requires MLX-formatted DeepSeek-OCR v2 weights. |
| **GLM-OCR** | `backend="glm-ocr"` | MPS (MLX) | Inline (equations embedded in OCR output) | Lightweight 0.5B VLM on Apple Silicon. |
| **OlmOCR-2** | `backend="olmocr"` | CUDA (vLLM) / MPS (MLX) | Inline (equations embedded in OCR output) | High-accuracy VLM-based OCR. |
| **MinerU** | `backend="mineru"` | CUDA / MPS / CPU | Inline (equations embedded in OCR output) | Wraps the external `magic-pdf` CLI. |

## Unified CLI

GlossAPI ships a Typer-based CLI with two subcommands:

```bash
glossapi setup      # Interactive environment provisioning wizard
glossapi pipeline   # Interactive phase-selection wizard with gum
glossapi            # No subcommand → launches the pipeline wizard
```

**Pipeline wizard** — Arrow-key driven phase selection with checkboxes, per-phase confirmation, optional JSONL export. Preset profiles: "Lightweight PDF smoke test", "MinerU demo", "Custom".

**Setup wizard** — Detects OS and available Python versions, prompts for mode (vanilla / rapidocr / mineru / deepseek / deepseek-ocr-2 / glm-ocr / olmocr), virtualenv path, optional model downloads, and post-install tests.

**Prerequisite:** Install [gum](https://github.com/charmbracelet/gum) for the interactive prompts (the CLI falls back to plain TTY prompts if gum is unavailable).

### One-command runner

For a single command that does setup + env activation + pipeline launch:

```bash
./scripts/glossapi-cli.sh
```

This script provisions the selected profile, sources the virtualenv, and launches `glossapi pipeline`. It prompts for the profile interactively unless you set `MODE=rapidocr` (or another mode) beforehand.

## Choose Your Install Path

| Scenario | Commands | Notes |
| --- | --- | --- |
| Pip users | `pip install glossapi` | Fast vanilla evaluation with minimal dependencies. |
| Mode automation (recommended) | `./dependency_setup/setup_glossapi.sh --mode {vanilla\|rapidocr\|deepseek\|deepseek-ocr-2\|glm-ocr\|olmocr\|mineru}` | Creates an isolated venv per mode, installs Rust crates, and can run the relevant pytest subset. |
| Manual editable install | `pip install -e .` after cloning | Keep this if you prefer to manage dependencies by hand. |
| Optional extras | `pip install glossapi[rapidocr]` / `[cuda]` / `[deepseek]` / `[docs]` | Install specific optional dependency groups only. |

See `docs/index.md` for detailed environment notes, CUDA/ORT combinations, and troubleshooting tips.

## Automated Environment Profiles

Use `dependency_setup/setup_glossapi.sh` to provision a virtualenv with the right dependency stack:

```bash
# Vanilla pipeline (no GPU OCR extras)
./dependency_setup/setup_glossapi.sh --mode vanilla --venv dependency_setup/.venvs/vanilla --run-tests

# Docling + RapidOCR mode
./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests

# DeepSeek OCR mode (requires weights under $GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr)
./dependency_setup/setup_glossapi.sh \
  --mode deepseek \
  --venv dependency_setup/.venvs/deepseek \
  --weights-root /path/to/model_weights \
  --run-tests --smoke-test

# DeepSeek OCR v2 mode (MLX/MPS, macOS)
./dependency_setup/setup_glossapi.sh \
  --mode deepseek-ocr-2 \
  --venv dependency_setup/.venvs/deepseek-ocr-2 \
  --download-deepseek-ocr2 \
  --run-tests

# MinerU OCR mode (uses external magic-pdf CLI)
./dependency_setup/setup_glossapi.sh \
  --mode mineru \
  --venv dependency_setup/.venvs/mineru \
  --download-mineru-models \
  --run-tests

# GLM-OCR mode (MLX/MPS, macOS Apple Silicon)
./dependency_setup/setup_glossapi.sh \
  --mode glm-ocr \
  --venv dependency_setup/.venvs/glm-ocr \
  --download-glmocr \
  --run-tests

# OlmOCR-2 mode (CUDA or MLX/MPS)
./dependency_setup/setup_glossapi.sh \
  --mode olmocr \
  --venv dependency_setup/.venvs/olmocr \
  --download-olmocr \
  --run-tests
```

The setup script auto-detects Python (preferring 3.12 → 3.11 → 3.13), installs Rust extensions in editable mode, and supports `--run-tests` / `--smoke-test` for post-install validation. Check `dependency_setup/dependency_notes.md` for the latest pins and caveats.

Pass `--download-deepseek` to fetch DeepSeek weights automatically; otherwise set `GLOSSAPI_WEIGHTS_ROOT` so the pipeline finds weights at `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr`.

Pass `--download-deepseek-ocr2` to fetch DeepSeek OCR v2 weights from `mlx-community/DeepSeek-OCR-2-8bit` into `$GLOSSAPI_WEIGHTS_ROOT/deepseek-ocr-mlx`.

Pass `--download-glmocr` to fetch GLM-OCR MLX weights from `mlx-community/GLM-OCR-4bit` into `$GLOSSAPI_WEIGHTS_ROOT/glm-ocr-mlx`.

Pass `--download-olmocr` to fetch OlmOCR weights into `$GLOSSAPI_WEIGHTS_ROOT/olmocr`.

<details>
<summary><strong>DeepSeek runtime checklist</strong></summary>

- Run `python -m glossapi.ocr.deepseek.preflight` (from your DeepSeek venv) to fail fast if the CLI would fall back to the stub.
- Export these to force the real CLI and avoid silent stub output:
  - `GLOSSAPI_DEEPSEEK_ALLOW_CLI=1`
  - `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0`
  - `GLOSSAPI_DEEPSEEK_VLLM_SCRIPT=/path/to/deepseek-ocr/run_pdf_ocr_vllm.py`
  - `GLOSSAPI_DEEPSEEK_TEST_PYTHON=/path/to/deepseek/venv/bin/python`
  - `GLOSSAPI_DEEPSEEK_MODEL_DIR=/path/to/model_weights/deepseek-ocr` (or set `GLOSSAPI_WEIGHTS_ROOT`)
  - `GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib`
- CUDA toolkit with `nvcc` available (FlashInfer/vLLM JIT falls back poorly without it); set `CUDA_HOME` and prepend `$CUDA_HOME/bin` to `PATH`.
- If FlashInfer is problematic, disable with `VLLM_USE_FLASHINFER=0` and `FLASHINFER_DISABLE=1`.
- To avoid FP8 KV cache issues, export `GLOSSAPI_DEEPSEEK_NO_FP8_KV=1` (propagates `--no-fp8-kv`).
- Tune VRAM use via `GLOSSAPI_DEEPSEEK_GPU_MEMORY_UTILIZATION=<0.5–0.9>`.

</details>

<details>
<summary><strong>MinerU runtime checklist</strong></summary>

- Ensure `magic-pdf` is on PATH (or pass `--mineru-command /path/to/magic-pdf` in setup).
- Prefer Python 3.11 for the MinerU venv (3.10–3.13 supported upstream).
- MinerU's layout module imports Detectron2. If Detectron2 is unavailable (common on macOS), the setup script auto-enables stub fallback. To install Detectron2 when you have a wheel, set `DETECTRON2_WHL_URL` before running `setup_glossapi.sh`.
- Export these to force the real CLI and avoid stub output:
  - `GLOSSAPI_MINERU_ALLOW_CLI=1`
  - `GLOSSAPI_MINERU_ALLOW_STUB=0`
  - `GLOSSAPI_MINERU_COMMAND=/path/to/magic-pdf` (optional override)
  - `GLOSSAPI_MINERU_MODE=auto` (or `fast`/`accurate` if your MinerU build supports it)

</details>

### macOS (Metal/MPS) quickstart

GlossAPI supports macOS GPU acceleration via Metal/MPS. Use the MinerU profile:

```bash
./dependency_setup/setup_glossapi.sh --mode mineru --venv dependency_setup/.venvs/mineru --download-mineru-models
source dependency_setup/.venvs/mineru/bin/activate
glossapi pipeline
```

See `docs/ocr_and_math_enhancement.md` for details.

## Rust Extensions

GlossAPI ships two Rust crates that are built as Python extensions via [maturin](https://github.com/PyO3/maturin) + [PyO3](https://pyo3.rs/):

| Crate | Location | Purpose |
| --- | --- | --- |
| `glossapi_rs_cleaner` | `rust/glossapi_rs_cleaner/` | Parallel markdown cleaning, mojibake detection, script filtering (Rayon + regex + Aho-Corasick). Outputs a Parquet report. |
| `glossapi_rs_noise` | `rust/glossapi_rs_noise/` | Noise-based markdown quality metrics (Rayon + memmap2). Integrated as the package-level native extension. |

Both crates target Python ≥ 3.8 via `abi3`. The `clean()` phase auto-builds `glossapi_rs_cleaner` with `maturin develop --release` if the module is not already available.

## Artifact Layout

A typical run produces the following directory tree under `output_dir/`:

```
output_dir/
├── downloads/               # Raw downloaded files
├── download_results/        # Download metadata parquet(s)
├── markdown/                # Phase-1 extracted Markdown
├── json/                    # Docling JSON + formula indexes (if export_doc_json=True)
├── clean_markdown/          # Phase-2 Rust-cleaned Markdown
├── sidecars/                # Per-file metadata (extract/, triage/, math/ subdirs)
├── sections/                # sections_for_annotation.parquet
├── logs/                    # Per-run log files
├── skiplists/               # Skiplist files for resume
├── export/                  # JSONL / sharded JSONL output
├── classified_sections.parquet
└── fully_annotated_sections.parquet
```

## Repo Landmarks

| Path | Description |
| --- | --- |
| `src/glossapi/` | Core library — Corpus orchestrator, phase mixins, OCR backends, Docling integration. |
| `src/glossapi/corpus/` | Phase implementations: `phase_download.py`, `phase_extract.py`, `phase_clean.py`, `phase_ocr_math.py`, `phase_sections.py`, `phase_annotate.py`, `phase_export.py`. |
| `src/glossapi/ocr/` | OCR backend wrappers: `rapidocr/`, `deepseek/`, `deepseek_ocr2/`, `glm_ocr/`, `olmocr/`, `mineru/`, `math/`, `utils/`. |
| `rust/` | Rust crates: `glossapi_rs_cleaner`, `glossapi_rs_noise`. |
| `samples/lightweight_pdf_corpus/` | Synthetic PDFs with manifest + expected outputs for smoke tests. |
| `tests/` | pytest suite — `test_pipeline_smoke.py` for quick checks, plus per-feature tests. |
| `docs/` | MkDocs Material site — onboarding, pipeline recipes, configuration, troubleshooting. |
| `dependency_setup/` | Environment provisioning scripts and notes per mode. |
| `scripts/glossapi-cli.sh` | One-command setup + pipeline launcher with interactive profile selection. |

## Contributing

- Run `pytest tests/test_pipeline_smoke.py` for a fast end-to-end check.
- Regenerate the lightweight corpus via `generate_pdfs.py` and commit the updated PDFs + manifest together.
- Prefer `uv` or `pip` editable installs so Rust extensions rebuild locally.
- New phases should be added as mixins under `src/glossapi/corpus/phase_*.py` and wired into `Corpus`.
- Respect the skiplist and metadata parquet contracts — see `corpus_skiplist.py` and `parquet_schema.py`.

Open an issue or PR if you spot drift between expected outputs and the pipeline, or if you have doc updates.

## Documentation

Build the docs locally:

```bash
pip install glossapi[docs]
mkdocs serve
```

The site is powered by [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and covers onboarding, pipeline details, OCR configuration, multi-GPU usage, AWS distribution, and troubleshooting.

## License

This project is licensed under the [EUPL 1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).
