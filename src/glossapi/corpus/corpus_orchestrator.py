from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .._naming import canonical_stem
from ..gloss_section import GlossSection
try:
    from ..gloss_section_classifier import GlossSectionClassifier  # type: ignore
except Exception:
    GlossSectionClassifier = None  # type: ignore[assignment]
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch
from .phase_download import DownloadPhaseMixin
from .phase_extract import ExtractPhaseMixin
from .phase_clean import CleanPhaseMixin
from .phase_ocr_math import OcrMathPhaseMixin
from .phase_sections import SectionPhaseMixin
from .phase_annotate import AnnotatePhaseMixin
from .phase_export import ExportPhaseMixin

class Corpus(
    DownloadPhaseMixin,
    ExtractPhaseMixin,
    CleanPhaseMixin,
    OcrMathPhaseMixin,
    SectionPhaseMixin,
    AnnotatePhaseMixin,
    ExportPhaseMixin,
):
    """
    A high-level wrapper for the GlossAPI academic document processing pipeline.
    
    This class provides a unified interface to extract PDFs to markdown,
    extract sections, and classify them using machine learning.
    
    Example:
        corpus = Corpus(input_dir="path/to/pdfs", output_dir="path/to/output")
        corpus.extract()  # Extract PDFs to markdown
        corpus.section()  # Extract sections from markdown files
        corpus.annotate()  # Classify sections using ML
    """

    
    def __init__(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path],
        section_classifier_model_path: Optional[Union[str, Path]] = None,
        extraction_model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        annotation_mapping: Optional[Dict[str, str]] = None,
        downloader_config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
        verbose: bool = False
    ):
        """
        Initialize the Corpus processor.
        
        Args:
            input_dir: Directory containing input files (PDF or markdown)
            output_dir: Base directory for all outputs
            section_classifier_model_path: Path to the pre-trained section classifier model
            extraction_model_path: Path to the pre-trained kmeans clustering model for extraction
            metadata_path: Path to metadata file with document types (optional)
            annotation_mapping: Dictionary mapping document types to annotation methods (optional)
                               e.g. {'Κεφάλαιο': 'chapter'} means documents with type 'Κεφάλαιο' use chapter annotation
            downloader_config: Configuration parameters for the GlossDownloader (optional)
            log_level: Logging level (default: logging.INFO)
            verbose: Whether to enable verbose logging for debugging (default: False)
        """
        # Verbose flag for detailed logging
        self.verbose = verbose
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._metadata_parquet_path: Optional[Path] = None

        self._setup_logging(log_level if not verbose else logging.DEBUG)
        self._resolve_model_paths(section_classifier_model_path, extraction_model_path)

        self.metadata_path = Path(metadata_path) if metadata_path else None
        if self.metadata_path is not None:
            try:
                if self.metadata_path.exists():
                    self._metadata_parquet_path = self.metadata_path
                else:
                    self.logger.warning("Provided metadata_path does not exist: %s", self.metadata_path)
                    self.metadata_path = None
            except Exception:
                self.metadata_path = None

        self.annotation_mapping = annotation_mapping or {'Κεφάλαιο': 'chapter'}
        self.downloader_config = downloader_config or {}
        self.url_column = self.downloader_config.get('url_column', 'url')

        # Lazy-create extractor to avoid heavy imports unless needed
        self.extractor = None
        self.sectioner = GlossSection()
        try:
            self.classifier = GlossSectionClassifier() if GlossSectionClassifier is not None else None  # type: ignore[call-arg, assignment]
        except Exception as exc:
            self.logger.debug("GlossSectionClassifier init failed (model/deps unavailable): %s", exc)
            self.classifier = None

        self._gpu_banner_logged = False
        self._phase1_backend = "safe"

        self._create_output_dirs()

        # Performance & Power Profiler — lazily imported to keep vanilla install lightweight.
        try:
            from glossapi.perf_metrics import PipelineProfiler as _PipelineProfiler
            self._profiler = _PipelineProfiler(output_dir=self.output_dir)
        except Exception:
            self._profiler = None  # type: ignore[assignment]
        self._perf_last_reported_count: int = 0

        self.sections_parquet = self.sections_dir / "sections_for_annotation.parquet"
        self.classified_parquet = self.output_dir / "classified_sections.parquet"
        self.fully_annotated_parquet = self.output_dir / "fully_annotated_sections.parquet"
        self.filename_to_doctype = {}

        self._load_metadata()

    # ------------------------------------------------------------------
    # __init__ helpers
    # ------------------------------------------------------------------

    def _setup_logging(self, log_level: int) -> None:
        """Configure the instance logger without touching global logging state."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        try:
            if not self.logger.handlers:
                _handler = logging.StreamHandler()
                _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s", "%H:%M:%S"))
                self.logger.addHandler(_handler)
            self.logger.propagate = False
        except Exception:
            pass

    def _resolve_model_paths(
        self,
        section_classifier_model_path: Optional[Union[str, Path]],
        extraction_model_path: Optional[Union[str, Path]],
    ) -> None:
        """Resolve and store section classifier and extraction model paths."""
        models_dir = Path(__file__).resolve().parent.parent / "models"
        self.section_classifier_model_path = (
            Path(section_classifier_model_path)
            if section_classifier_model_path
            else models_dir / "section_classifier.joblib"
        )
        self.extraction_model_path = (
            Path(extraction_model_path)
            if extraction_model_path
            else models_dir / "kmeans_weights.joblib"
        )

    def _create_output_dirs(self) -> None:
        """Create all required output sub-directories under ``self.output_dir``."""
        self.downloads_dir = self.output_dir / "downloads"
        self.markdown_dir = self.output_dir / "markdown"
        self.sections_dir = self.output_dir / "sections"
        self.cleaned_markdown_dir = self.output_dir / "clean_markdown"
        # models_dir is defined here but created on demand (not guaranteed to exist)
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        os.makedirs(self.cleaned_markdown_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Performance & Power Metrics
    # ------------------------------------------------------------------

    def perf_report(self, *, backend: Optional[str] = None) -> Dict[str, Any]:
        """Generate and save a Performance & Power Report for all recorded phases.

        Returns a structured dict with per-phase and end-to-end metrics::

            {
              "run_id": "...",
              "backend": "deepseek-ocr",
              "total_pages": 120,
              "power_source": "nvml",       # or 'rapl' / 'unavailable'
              "phases": {
                "extract": {"active_sec": 45.2, "pps": 2.65, "ppw": 0.012, ...},
                "ocr":     {"active_sec": 210.5, "pps": 0.57, ...},
                "clean":   {"active_sec": 3.1,   "pps": 38.7, ...}
              },
              "end_to_end": {"pps": 0.46, "ppw": 0.009, ...}
            }

        The report JSON is also saved to
        ``output_dir/logs/perf_report_<backend>_<timestamp>.json``.

        Call this after running one or more pipeline phases::

            corpus.extract()
            corpus.clean()
            corpus.ocr(backend="deepseek-ocr")
            report = corpus.perf_report(backend="deepseek-ocr")
        """
        if self._profiler is None:
            self.logger.warning("perf_report(): profiler not initialised (perf_metrics unavailable).")
            return {}
        try:
            return self._profiler.report(backend=backend)
        except Exception as exc:
            self.logger.warning("perf_report() failed: %s", exc)
            return {}

    def reset_perf_metrics(self) -> None:
        """Clear all accumulated phase samples and start fresh.

        Call this before re-running the pipeline on a different dataset to
        avoid mixing measurements from separate runs.
        """
        if self._profiler is not None:
            self._profiler.reset()
        self._perf_last_reported_count = 0

    def _maybe_emit_perf_report(self, backend: Optional[str] = None) -> None:
        """Log a compact snapshot for the most recently completed phase.

        Provides per-phase progress visibility without writing a full JSON
        report after every phase.  The full JSON report is only written when
        :py:meth:`perf_report` is called explicitly (or by the pipeline wizard)
        at the end of the full pipeline sequence.
        """
        if self._profiler is None:
            return
        try:
            self._profiler._log_phase_snapshot()
        except Exception:
            pass

    def _get_cached_metadata_parquet(self) -> Optional[Path]:
        """Return cached metadata parquet path if it still exists."""

        if self._metadata_parquet_path is not None:
            if self._metadata_parquet_path.exists():
                return self._metadata_parquet_path
            self._metadata_parquet_path = None
        return None

    def _cache_metadata_parquet(self, candidate: Optional[Union[str, Path]]) -> Optional[Path]:
        """Remember the provided parquet path for subsequent lookups."""

        if candidate is None:
            return None
        path = Path(candidate)
        self._metadata_parquet_path = path
        return path

    def _resolve_metadata_parquet(
        self,
        parquet_schema: "ParquetSchema",
        *,
        ensure: bool = True,
        search_input: bool = True,
    ) -> Optional[Path]:
        """Return a best-effort metadata parquet path, caching the result."""

        cached = self._get_cached_metadata_parquet()
        if cached is not None:
            return cached
        if ensure:
            ensured = parquet_schema.ensure_metadata_parquet(self.output_dir)
            if ensured is not None:
                return self._cache_metadata_parquet(ensured)
        found = parquet_schema.find_metadata_parquet(self.output_dir)
        if found is not None:
            return self._cache_metadata_parquet(found)
        if search_input:
            input_dirs = [self.input_dir]
            dl_dir = self.input_dir / "download_results"
            if dl_dir.exists():
                input_dirs.append(dl_dir)
            for directory in input_dirs:
                if directory.exists():
                    located = parquet_schema.find_metadata_parquet(directory)
                    if located is not None:
                        return self._cache_metadata_parquet(located)
        return None

    # ------------------------------------------------------------------
    # Shared pipeline helpers (available to all phase mixins via self)
    # ------------------------------------------------------------------

    @functools.cached_property
    def _parquet_schema(self) -> "ParquetSchema":  # type: ignore[name-defined]
        """Shared ParquetSchema instance, keyed by this corpus's url_column."""
        from glossapi.parquet_schema import ParquetSchema
        return ParquetSchema({"url_column": self.url_column})

    def _get_skip_manager(self) -> "_SkiplistManager":
        """Return a fresh skip-list manager backed by this run's skiplist file."""
        path = _resolve_skiplist_path(self.output_dir, self.logger)
        return _SkiplistManager(path, self.logger)

    def _load_metadata(self) -> None:
        """Load the metadata parquet and build a *canonical stem \u2192 document type* mapping."""
        if not (self.metadata_path and self.metadata_path.exists()):
            if self.metadata_path:
                self.logger.warning("Metadata file not found: %s", self.metadata_path)
            return

        try:
            metadata_df = pd.read_parquet(self.metadata_path)
            self.logger.info(
                "Loading metadata from %s (%d rows, columns: %s)",
                self.metadata_path,
                len(metadata_df),
                metadata_df.columns.tolist(),
            )

            if "document_type" not in metadata_df.columns:
                metadata_df["document_type"] = pd.NA
                self.logger.info("Added missing 'document_type' column to metadata")
            if "filename" not in metadata_df.columns:
                self.logger.warning("Metadata file is missing a 'filename' column; skipping mapping")
                return

            # Drop rows where document_type is absent or blank
            mask = metadata_df["document_type"].notna() & (
                metadata_df["document_type"].astype(str).str.strip() != ""
            )
            df_valid = metadata_df[mask]

            # canonical_stem normalises any extension variant, so a single key
            # per document matches regardless of .pdf / .md / no-extension form.
            self.filename_to_doctype = {
                canonical_stem(fn): dt
                for fn, dt in zip(
                    df_valid["filename"].astype(str),
                    df_valid["document_type"].astype(str),
                )
                if fn
            }
            self.logger.info(
                "Loaded %d filename-to-doctype mappings", len(self.filename_to_doctype)
            )
        except Exception as exc:
            self.logger.error("Error loading metadata from %s: %s", self.metadata_path, exc)

    # All phase logic lives in the respective PhaseMixin classes inherited above.


# Top-level worker function for multi-GPU extraction (picklable by multiprocessing)
def gpu_extract_worker_queue(
    device_id: int,
    in_dir: str,
    out_dir: str,
    work_q,  # multiprocessing Queue of filename strings
    force: bool,
    fe: bool,
    ce: bool,
    use_cls_w: bool,
    skip: bool,
    input_fmt: str,
    threads: int,
    benchmark: bool,
    export_json: bool,
    emit_index: bool,
    backend: str,
    result_q=None,
    status_map=None,
    marker_dir: Optional[str] = None,
) -> None:
    import os as _os
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    def _ensure_thread_caps():
        caps = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        for k, v in caps.items():
            _os.environ.setdefault(k, v)
        try:
            _torch = _maybe_import_torch()
            if _torch is not None and hasattr(_torch, "set_num_threads"):
                _torch.set_num_threads(1)
        except Exception:
            pass

    _ensure_thread_caps()
    _status_proxy = status_map
    _marker_path = _Path(marker_dir).expanduser() / f"gpu{device_id}.current" if marker_dir else None

    def _update_current(batch_items: List[str]) -> None:
        if _status_proxy is not None:
            try:
                _status_proxy[device_id] = list(batch_items)
            except Exception:
                pass
        if _marker_path is not None:
            try:
                _marker_path.write_text("\n".join(batch_items) + "\n", encoding="utf-8")
            except Exception:
                pass

    def _clear_current() -> None:
        if _status_proxy is not None:
            try:
                _status_proxy.pop(device_id, None)
            except Exception:
                pass
        if _marker_path is not None:
            try:
                _marker_path.unlink(missing_ok=True)
            except Exception:
                pass
    _worker_log_handle = None
    try:
        _log_dir = _os.environ.get("GLOSSAPI_WORKER_LOG_DIR")
        if _log_dir:
            _log_path = _Path(_log_dir).expanduser()
            _log_path.mkdir(parents=True, exist_ok=True)
            _worker_log_file = _log_path / f"gpu{device_id}_{_os.getpid()}.log"
            _worker_log_handle = open(_worker_log_file, "a", encoding="utf-8", buffering=1)
            _sys.stdout = _worker_log_handle
            _sys.stderr = _worker_log_handle
    except Exception:
        _worker_log_handle = None
    # Bind this worker to a single GPU id
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _os.environ["GLOSSAPI_DOCLING_DEVICE"] = "cuda:0"
    # Worker GPU binding banner (prints by default; disable with GLOSSAPI_WORKER_LOG_VERBOSE=0)
    try:
        _verbose = str(_os.environ.get("GLOSSAPI_WORKER_LOG_VERBOSE", "1")).strip().lower()
        if _verbose not in ("0", "false", "no", "off", ""):  # default on
            try:
                _torch = _maybe_import_torch()
                if _torch is not None and getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                    _torch_name = _torch.cuda.get_device_name(0)
                elif _torch is not None:
                    _torch_name = "no-cuda"
                else:
                    _torch_name = "unloaded"
            except Exception:
                _torch_name = "unknown"
            try:
                import onnxruntime as _ort  # type: ignore
                _ort_prov = _ort.get_available_providers()
            except Exception:
                _ort_prov = []
            try:
                import subprocess as _sp
                _nvsmi = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
                _phys = _nvsmi.stdout.splitlines()[0].strip() if _nvsmi.returncode == 0 and _nvsmi.stdout else ""
            except Exception:
                _phys = ""
            try:
                print(f"[GPU{device_id}] bound: CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES','')} pid={_os.getpid()} torch={_torch_name} ORT={_ort_prov}")
                if _phys:
                    print(f"[GPU{device_id}] physical: {_phys}")
            except Exception:
                pass
    except Exception:
        pass
    # Light import of Corpus (prefer installed package; fallback to repo src)
    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            import sys as _sys, pathlib as _pl
            _sys.path.insert(0, str((_pl.Path(out_dir).resolve().parents[1] / 'src').resolve()))
            _ensure_thread_caps()
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as _e:
            print(f"[GPU{device_id}] Cannot import glossapi in worker: {_e}")
            if result_q is not None:
                try:
                    result_q.put(
                        {
                            "event": "exit",
                            "worker": device_id,
                            "exitcode": 1,
                            "pid": _os.getpid(),
                            "error": str(_e),
                        }
                    )
                except Exception:
                    pass
            _sys.exit(1)
    c = _Corpus(input_dir=in_dir, output_dir=out_dir)
    # Prime once per worker (persistent converter)
    try:
        c.prime_extractor(
            input_format=input_fmt,
            num_threads=threads,
            accel_type="cuda:0",
            force_ocr=force,
            formula_enrichment=fe,
            code_enrichment=ce,
            use_cls=use_cls_w,
            benchmark_mode=benchmark,
            export_doc_json=bool(export_json),
            emit_formula_index=bool(emit_index),
            phase1_backend=backend,
        )
    except Exception as _e:
        msg = f"[GPU{device_id}] Prime failed: {_e}"
        print(msg)
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "exit",
                        "worker": device_id,
                        "exitcode": 1,
                        "pid": _os.getpid(),
                        "error": str(_e),
                    }
                )
            except Exception:
                pass
        raise
    try:
        if c.extractor is not None:
            c.extractor.external_state_updates = result_q is not None

            if result_q is not None:

                def _report_batch(ok_list, bad_list):
                    try:
                        result_q.put(
                            {
                                "event": "batch",
                                "worker": device_id,
                                "processed": [str(x) for x in ok_list],
                                "problematic": [str(x) for x in bad_list],
                                "pid": _os.getpid(),
                            }
                        )
                    except Exception as exc:
                        print(f"[GPU{device_id}] Failed to report batch: {exc}")

                c.extractor.batch_result_callback = _report_batch
    except Exception as _e:
        print(f"[GPU{device_id}] Unable to set batch callback: {_e}")
    # Prepare persistent extractor in this worker on first call
    # Process queue items in small batches to reduce function-call overhead
    batch: list[str] = []
    try:
        _batch_env = int(str(_os.environ.get("GLOSSAPI_GPU_BATCH_SIZE", "")).strip() or 0)
    except Exception:
        _batch_env = 0
    default_batch = 5 if not force else 1
    try:
        extractor = getattr(c, "extractor", None)
        if extractor is not None:
            configured = int(getattr(extractor, "max_batch_files", default_batch))
            if force:
                default_batch = 1
            else:
                default_batch = max(1, configured)
    except Exception:
        pass
    BATCH_SIZE = max(1, _batch_env) if _batch_env else max(1, default_batch)
    import queue as _queue
    last_progress = _time.time()
    processed = 0
    exit_code = 0
    try:
        while True:
            try:
                nm = work_q.get_nowait()
            except _queue.Empty:
                # queue.Empty or other -> flush any pending batch then exit
                if batch:
                    try:
                        _update_current(list(batch))
                        c.extract(
                            input_format=input_fmt,
                            num_threads=threads,
                            accel_type="cuda:0",
                            force_ocr=force,
                            formula_enrichment=fe,
                            code_enrichment=ce,
                            file_paths=list(batch),
                            skip_existing=skip,
                            use_gpus="single",
                            use_cls=use_cls_w,
                            benchmark_mode=benchmark,
                            export_doc_json=bool(export_json),
                            emit_formula_index=bool(emit_index),
                            phase1_backend=backend,
                            _prepared=True,
                        )
                        processed += len(batch)
                        _clear_current()
                    except Exception as _e:
                        exit_code = 1
                        print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
                        if result_q is not None:
                            try:
                                result_q.put(
                                    {
                                        "event": "batch",
                                        "worker": device_id,
                                        "processed": [],
                                        "problematic": list(batch),
                                        "pid": _os.getpid(),
                                        "error": str(_e),
                                    }
                                )
                            except Exception:
                                pass
                        _clear_current()
                    batch.clear()
                break
            except Exception as exc:
                exit_code = 1
                print(f"[GPU{device_id}] Queue receive error: {exc}")
                break
            if isinstance(nm, str) and nm.strip():
                batch.append(nm)
            if len(batch) >= BATCH_SIZE:
                try:
                    _update_current(list(batch))
                    c.extract(
                        input_format=input_fmt,
                        num_threads=threads,
                        accel_type="cuda:0",
                        force_ocr=force,
                        formula_enrichment=fe,
                        code_enrichment=ce,
                        file_paths=list(batch),
                        skip_existing=skip,
                        use_gpus="single",
                        use_cls=use_cls_w,
                        benchmark_mode=benchmark,
                        export_doc_json=bool(export_json),
                        emit_formula_index=bool(emit_index),
                        phase1_backend=backend,
                        _prepared=True,
                    )
                    processed += len(batch)
                    _clear_current()
                except Exception as _e:
                    exit_code = 1
                    print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
                    if result_q is not None:
                        try:
                            result_q.put(
                                {
                                    "event": "batch",
                                    "worker": device_id,
                                    "processed": [],
                                    "problematic": list(batch),
                                    "pid": _os.getpid(),
                                    "error": str(_e),
                                }
                            )
                        except Exception:
                            pass
                    _clear_current()
                batch.clear()
            # Occasional heartbeat
            if _time.time() - last_progress > 30:
                try:
                    print(f"[GPU{device_id}] processed ~{processed} files…")
                except Exception:
                    pass
                last_progress = _time.time()
    except Exception as exc:
        exit_code = 1
        print(f"[GPU{device_id}] Fatal worker error: {exc}")

    _clear_current()

    try:
        extractor = getattr(c, "extractor", None)
        release = getattr(extractor, "release_resources", None)
        if callable(release):
            release()
    except Exception as exc:
        try:
            print(f"[GPU{device_id}] Failed to release extractor resources: {exc}")
        except Exception:
            pass

    if result_q is not None:
        try:
            result_q.put({
                "event": "exit",
                "worker": device_id,
                "exitcode": exit_code,
                "pid": _os.getpid(),
            })
        except Exception as exc:
            print(f"[GPU{device_id}] Failed to report exit: {exc}")

    if _worker_log_handle is not None:
        try:
            _worker_log_handle.flush()
            _worker_log_handle.close()
        except Exception:
            pass
    _sys.exit(exit_code)
