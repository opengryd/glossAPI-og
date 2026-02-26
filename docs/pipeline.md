# Pipeline Overview & Artifacts

GlossAPI is a staged pipeline. You can enter at any stage and use the same folder for input and output.

## Corpus usage contract

The `Corpus` class is the stable surface of the project. New functionality should plug into the existing phase mixins so callers can stick to the small set of entrypoints (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `export/jsonl*()`). The expected usage pattern is a short script that chains these calls; avoid ad-hoc monkeypatches or bypassing the orchestrator when adding features so downstream users retain resumability and consistent artifacts.

## Stages

| Phase | Method | Description |
|---|---|---|
| Download | `corpus.download()` | Fetch PDFs from a URL parquet (resume-aware, parallel scheduler grouping). |
| Extract | `corpus.extract()` | Convert documents to Markdown. Backends: `"safe"` (PyPDFium), `"docling"`, or `"auto"`. Supports PDF, DOCX, HTML, XML/JATS, PPTX, CSV, MD. |
| Clean | `corpus.clean()` | Rust-powered cleaning and mojibake detection via `glossapi_rs_cleaner` + quality scoring via `glossapi_rs_noise`. Sets `needs_ocr` flag in metadata. |
| OCR / Math | `corpus.ocr()` | Re-OCR bad documents and/or enrich math. Backends: `"rapidocr"`, `"deepseek-ocr"`, `"deepseek-ocr-2"`, `"glm-ocr"`, `"mineru"`, `"olmocr"`. |
| Section | `corpus.section()` | Extract sections from Markdown into a structured Parquet. |
| Annotate | `corpus.annotate()` | Classify sections with a pre-trained model. Modes: `"text"`, `"chapter"`, `"auto"`. |
| Export | `corpus.jsonl()` / `corpus.jsonl_sharded()` | Produce JSONL (optionally zstd-compressed shards) with merged metadata. |

A convenience method `corpus.process_all()` chains `extract → section → annotate` in one call (optionally `download()` first via `download_first=True`). It does not include `clean()` or `ocr()` steps.

## Artifact Layout

```
OUT/
├── downloads/               # Raw downloaded files
│   └── problematic_math/    # Quarantined PDFs (respawn cap exceeded)
├── download_results/        # Download metadata parquet(s)
├── markdown/                # Phase-1 extracted Markdown (overwritten by enrichment)
│   └── <stem>.md
├── json/                    # Docling JSON + formula indexes
│   ├── <stem>.docling.json(.zst)
│   ├── <stem>.formula_index.jsonl
│   ├── <stem>.latex_map.jsonl
│   ├── metrics/
│   │   ├── <stem>.metrics.json
│   │   └── <stem>.per_page.metrics.json
│   └── problematic_math/    # Quarantined Docling artifacts
├── clean_markdown/          # Rust-cleaned Markdown (phase clean output)
├── sidecars/                # Per-file metadata
│   ├── extract/             # Extraction metadata
│   ├── triage/              # Formula density / OCR routing
│   └── math/                # Math enrichment metadata
├── sections/                # sections_for_annotation.parquet
├── logs/                    # Per-run log files
│   ├── ocr_workers/         # Per-GPU OCR worker logs
│   └── math_workers/        # Per-GPU math worker logs + gpu<N>.current
├── skiplists/               # fatal_skip.txt + phase-specific skiplists
├── export/                  # JSONL / sharded JSONL output (.jsonl.zst)
├── classified_sections.parquet
└── fully_annotated_sections.parquet
```

Notes:
- Enriched Markdown replaces the plain Markdown (single canonical location).
- Metrics lived under `markdown/` in earlier versions; they now live under `json/metrics/`.
- When math enrichment cannot recover after the configured number of respawns, the corresponding PDFs and Docling artifacts are copied into the `problematic_math/` folders above and the stems are added to the fatal skip-list for later review.

## Exporting corpora

Use `Corpus.jsonl(...)` when you want a single JSONL file (e.g. quick inspection) and `Corpus.jsonl_sharded(...)` when preparing pretraining releases. Both calls accept the same knobs for renaming the text column, nesting pipeline metadata, and wiring an external source-metadata parquet.

```python
from pathlib import Path
from glossapi import Corpus

corpus = Corpus(input_dir=Path("input"), output_dir=Path("out"))

shards = corpus.jsonl_sharded(
    Path("out/export"),
    shard_size_bytes=500 * 1024 * 1024,
    shard_prefix="train",
    text_key="text",
    metadata_key="pipeline_metadata",
    metadata_fields=[
        "filter",
        "greek_badness_score",
        "is_empty",
        "latin_percentage",
        "mojibake_badness_score",
        "needs_ocr",
        "percentage_greek",
        "polytonic_ratio",
    ],
    include_remaining_metadata=False,
    metadata_path=Path("out/download_results/didaktorika_downloads_enhanced.parquet"),
    source_metadata_key="source_metadata",
    source_metadata_fields=["filename", "language", "handle_url", "date_accepted"],
    source_metadata_path=Path("out/source_metadata/didaktorika_full_enriched_FINAL.parquet"),
)
```

### Loading in downstream trainers

Hugging Face Datasets can stream the resulting `.jsonl.zst` shards without unpacking:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="out/export/train-*.jsonl.zst",
    streaming=True,
)["train"]

for row in dataset:
    text = row["text"]  # pipeline metadata is under row["pipeline_metadata"]
    break
```

For analytics or training-time filtering, keep the Parquet sidecars keyed by `doc_id` (or filename) and use PyArrow predicates:

```python
import pyarrow.dataset as ds

dataset = ds.dataset("out/metadata/source_metadata.parquet", format="parquet")
recent_greek = dataset.to_table(
    filter=(ds.field("language") == "Ελληνικά") &
            (ds.field("date_accepted") >= "2018-01-01")
)
```

These snippets are mirrored in the test suite so regressions in file layout or compression settings are caught automatically.
