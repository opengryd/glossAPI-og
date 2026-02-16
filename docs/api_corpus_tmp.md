# API Reference — `glossapi.Corpus`

The `Corpus` class is the high‑level entrypoint for the pipeline. Below are the most commonly used methods.

## Constructor

```python
glossapi.Corpus(
  input_dir: str | Path,
  output_dir: str | Path,
  section_classifier_model_path: str | Path | None = None,
  extraction_model_path: str | Path | None = None,
  metadata_path: str | Path | None = None,
  annotation_mapping: dict[str, str] | None = None,
  downloader_config: dict[str, Any] | None = None,
  log_level: int = logging.INFO,
  verbose: bool = False,
)
```

- `input_dir`: source files (PDF/DOCX/HTML/…)
- `output_dir`: pipeline outputs (markdown, json, sections, …)
- `annotation_mapping`: defaults to `{'Κεφάλαιο': 'chapter'}` if not provided.
- `downloader_config`: defaults for `download()` (e.g., concurrency, cookies)

## download()

```python
download(
  input_parquet: str | Path | None = None,
  url_column: str = 'url',
  verbose: bool | None = None,
  *,
  parallelize_by: str | None = None,
  links_column: str | None = None,
  **kwargs,
) -> pd.DataFrame
```

- Concurrent downloader with per‑domain scheduler, retries, and checkpoints.
- `input_parquet` defaults to auto-discovery from `input_dir` if not given.
- `url_column` (default `'url'`) and `links_column` control which column(s) contain download URLs.
- Outputs `download_results/*.parquet` and files in `downloads/`.

## extract()

```python
extract(
  input_format: str = 'all',
  num_threads: int | None = None,
  accel_type: str = 'CUDA',        # 'CPU'|'CUDA'|'MPS'|'Auto'
  *,
  force_ocr: bool = False,
  formula_enrichment: bool = False,
  code_enrichment: bool = False,
  filenames: list[str] | None = None,
  file_paths: list[str | Path] | None = None,
  skip_existing: bool = True,
  use_gpus: str = 'single',        # 'single'|'multi'
  devices: list[int] | None = None,
  use_cls: bool = False,
  benchmark_mode: bool = False,
  export_doc_json: bool = True,
  emit_formula_index: bool = False,
  phase1_backend: str = 'auto',    # 'auto'|'safe'|'docling'
  _prepared: bool = False,
) -> None
```

- Phase‑1 extraction; set `force_ocr=True` for OCR.
- `file_paths`: absolute paths to specific files to extract (alternative to `filenames`).
- `phase1_backend`: backend selection — `'auto'` (default), `'safe'` (SafeRapidOcrModel), or `'docling'` (direct Docling).
- Docling layout JSON writes by default (`json/<stem>.docling.json(.zst)`); set `emit_formula_index=True` to also produce `json/<stem>.formula_index.jsonl`.
- Set `use_gpus='multi'` to use all visible GPUs (shared queue).
- `_prepared`: internal flag — do not set manually.

## clean()

```python
clean(
  input_dir: str | Path | None = None,
  threshold: float = 0.10,
  num_threads: int | None = None,
  drop_bad: bool = True,
  *,
  write_cleaned_files: bool = True,
  empty_char_threshold: int = 0,
  empty_min_pages: int = 0,
) -> None
```

- Runs the Rust cleaner/noise metrics and populates parquet with badness; sets `good_files` and points `markdown_dir` to cleaned files for downstream.
- `write_cleaned_files`: if `True` (default), writes cleaned Markdown to `clean_markdown/`.
- `empty_char_threshold` / `empty_min_pages`: control near-empty document detection thresholds.

## ocr()

```python
ocr(
  *,
  fix_bad: bool = True,
  mode: str | None = None,
  backend: str = 'rapidocr',
  mineru_backend: str | None = None,
  device: str | None = None,
  model_dir: str | Path | None = None,
  max_pages: int | None = None,
  persist_engine: bool = True,
  limit: int | None = None,
  dpi: int | None = None,
  precision: str | None = None,
  math_enhance: bool = True,
  math_targets: dict[str, list[tuple[int,int]]] | None = None,
  math_batch_size: int = 8,
  math_dpi_base: int = 220,
  use_gpus: str = 'single',
  devices: list[int] | None = None,
  force: bool | None = None,
  reprocess_completed: bool | None = None,
  skip_existing: bool | None = None,
  content_debug: bool = False,
  CONTENT_DEBUG: bool | None = None,
  internal_debug: bool = False,
  INTERNAL_DEBUG: bool | None = None,
) -> None
```

- All parameters are **keyword-only**.
- Convenience shim that re‑runs `extract(force_ocr=True)` on cleaner-flagged documents and, by default, performs math/code enrichment unless `math_enhance=False`.
- `backend`: one of `'rapidocr'` (default), `'deepseek'`, `'deepseek-ocr-2'`, `'mineru'`.
- `mineru_backend`: override for MinerU's internal backend selection.
- `mode`: explicit mode selection — `'ocr_bad'`, `'math_only'`, or `'ocr_bad_then_math'`. If omitted, inferred from `fix_bad`/`math_enhance`.
- `reprocess_completed`: re-process files that have already been marked as done.
- `skip_existing`: skip files with existing enriched output.
- `force`: deprecated alias for `fix_bad`.
- `content_debug` / `CONTENT_DEBUG`: enable content-level debug logging.
- `internal_debug` / `INTERNAL_DEBUG`: enable internal-state debug logging.

## formula_enrich_from_json()

```python
formula_enrich_from_json(
  files: list[str] | None = None,
  *,
  device: str = 'cuda',
  batch_size: int = 8,
  dpi_base: int = 220,
  targets_by_stem: dict[str, list[tuple[int,int]]] | None = None,
) -> None
```

- Phase‑2 enrichment from Docling JSON. Writes enriched MD into `markdown/<stem>.md`, and `json/<stem>.latex_map.jsonl`.

## section(), annotate()

```python
section() -> None
annotate(annotation_type: str = 'text', fully_annotate: bool = True) -> None
```

- `section()` builds `sections/sections_for_annotation.parquet`. Takes no parameters.
- `annotate()` classifies sections — `annotation_type` is `'text'` (title-based) or `'chapter'` (chapter-number-based). Saves `classified_sections.parquet` and `fully_annotated_sections.parquet`.

## jsonl()

```python
jsonl(
  output_path: str | Path,
  *,
  text_key: str = 'document',
  metadata_key: str | None = None,
  metadata_fields: Iterable[str] | None = None,
  include_remaining_metadata: bool = True,
  metadata_path: str | Path | None = None,
  source_metadata_key: str | None = None,
  source_metadata_fields: Iterable[str] | None = None,
  source_metadata_path: str | Path | None = None,
) -> Path
```

- Exports all pipeline output to a single JSONL file.
- `text_key`: key for the main document text (default `'document'`).
- `metadata_key` / `metadata_fields`: control metadata embedding from pipeline Parquet.
- `source_metadata_*` params: attach provenance from an external Parquet file.

## jsonl_sharded()

```python
jsonl_sharded(
  output_dir: str | Path,
  *,
  shard_size_bytes: int = 500 * 1024 * 1024,   # 500 MiB
  shard_prefix: str = 'train',
  compression: str = 'zstd',
  compression_level: int = 3,
  text_key: str = 'document',
  metadata_key: str | None = None,
  metadata_fields: Iterable[str] | None = None,
  include_remaining_metadata: bool = True,
  metadata_path: str | Path | None = None,
  source_metadata_key: str | None = None,
  source_metadata_fields: Iterable[str] | None = None,
  source_metadata_path: str | Path | None = None,
) -> list[Path]
```

- Exports to sharded, optionally compressed JSONL files.
- `shard_size_bytes`: target shard size (default 500 MiB).
- `compression`: `'zstd'` (default) or `'none'`.
- Returns the list of created shard file paths.

## process_all()

```python
process_all(
  input_format: str = 'pdf',
  fully_annotate: bool = True,
  annotation_type: str = 'auto',
  download_first: bool = False,
) -> None
```

- Convenience shortcut that chains `extract → section → annotate` (no clean/OCR).
- Set `download_first=True` to run `download()` before extraction.

## triage_math()

Summarizes per‑page metrics and recommends Phase‑2 for math‑dense docs. Updates `download_results` parquet.

---

See also:

- [Configuration and environment variables](configuration.md)
- [OCR and math enrichment details](ocr_and_math_enhancement.md)
- [Math enrichment runtime guide](math_enrichment_runtime.md)
