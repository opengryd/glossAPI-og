"""Temporary wrappers around Docling's RapidOCR integration.

The upstream Docling release (2.48.x) does not tolerate RapidOCR returning
``None`` for a given crop. That bubbles up as an AttributeError inside the
conversion loop and the entire document fails. Until Docling includes a fix, we
wrap the loader so that ``None`` simply means "no detections" and processing
continues. Once Docling ships a release with the guard we can drop this shim and
revert to the vanilla ``RapidOcrModel``.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, Type

import numpy
from tqdm import tqdm

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, RapidOcrOptions
from docling.models.rapid_ocr_model import RapidOcrModel as _RapidOcrModel
from docling.models.rapid_ocr_model import TextCell, _log
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

from ._paths import resolve_packaged_onnx_and_keys

# ---------------------------------------------------------------------------
# Per-page OCR timeout (seconds).  Controlled by GLOSSAPI_OCR_PAGE_TIMEOUT.
# Default 60 s — should be enough for even large scanned pages with CoreML.
# ---------------------------------------------------------------------------
_DEFAULT_PAGE_TIMEOUT_S = 60

# ---------------------------------------------------------------------------
# OCR crop scale.  Docling hardcodes scale=3 (3×72 = 216 DPI).  On macOS
# Apple Silicon this makes full-page scans enormous (~7400×10500 px) and
# very slow.  A scale of 2 (144 DPI) is sufficient for PaddleOCR and cuts
# image area by ~55%.  Override via GLOSSAPI_OCR_CROP_SCALE.
# ---------------------------------------------------------------------------
_DEFAULT_OCR_CROP_SCALE = 2


def _get_page_timeout() -> float:
    """Return the per-page OCR timeout in seconds from env or default."""
    try:
        return float(os.getenv("GLOSSAPI_OCR_PAGE_TIMEOUT", str(_DEFAULT_PAGE_TIMEOUT_S)))
    except (ValueError, TypeError):
        return _DEFAULT_PAGE_TIMEOUT_S


def _get_ocr_crop_scale() -> float:
    """Return the OCR crop scale factor from env or default."""
    try:
        v = float(os.getenv("GLOSSAPI_OCR_CROP_SCALE", str(_DEFAULT_OCR_CROP_SCALE)))
        if v < 0.5 or v > 6:
            _log.warning("GLOSSAPI_OCR_CROP_SCALE=%s out of range [0.5, 6]; using default %s", v, _DEFAULT_OCR_CROP_SCALE)
            return float(_DEFAULT_OCR_CROP_SCALE)
        return v
    except (ValueError, TypeError):
        return float(_DEFAULT_OCR_CROP_SCALE)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _patch_onnx_sessions_coreml(reader: object) -> bool:
    """Patch RapidOCR ONNX sessions to use CoreMLExecutionProvider on macOS.

    RapidOCR's ``ProviderConfig`` only supports CPU/CUDA/DirectML/CANN providers.
    On macOS with Apple Silicon, we can leverage CoreML to accelerate OCR via the
    Apple Neural Engine / GPU.  This function replaces the underlying
    ``onnxruntime.InferenceSession`` objects with new ones that request
    ``CoreMLExecutionProvider`` as the primary provider.

    Returns True if at least one session was successfully patched.
    """
    if not _is_macos():
        return False

    try:
        import onnxruntime as ort
    except ImportError:
        return False

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in available:
        _log.info("CoreMLExecutionProvider not available; skipping ONNX session patching")
        return False

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    # CoreML session options: enable ANE + GPU compute units for best perf
    provider_options: list[dict] = [
        {"MLComputeUnits": "ALL"},  # Use ANE + GPU + CPU adaptively
        {},
    ]

    patched = 0
    # Attribute paths: reader.text_det.session.session, reader.text_cls.session.session,
    #                  reader.text_rec.session.session
    for component_name in ("text_det", "text_cls", "text_rec"):
        component = getattr(reader, component_name, None)
        if component is None:
            continue
        ort_wrapper = getattr(component, "session", None)
        if ort_wrapper is None:
            continue
        old_session = getattr(ort_wrapper, "session", None)
        if old_session is None:
            continue

        # Extract the model path from the existing session
        model_path = None
        try:
            model_path = old_session.get_modelmeta().graph_name
        except Exception:
            pass

        # Try to get the model bytes or path from the session
        # onnxruntime stores _model_path on the session object
        session_path = getattr(old_session, "_model_path", None)
        if not session_path:
            # Fallback: look at the OrtInferSession wrapper which stores model_path
            session_path = getattr(ort_wrapper, "model_path", None)

        if not session_path:
            _log.warning(
                "Cannot find model path for %s; skipping CoreML patch", component_name
            )
            continue

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            new_session = ort.InferenceSession(
                str(session_path),
                sess_options=sess_opts,
                providers=providers,
                provider_options=provider_options,
            )
            ort_wrapper.session = new_session
            actual_providers = new_session.get_providers()
            _log.info(
                "Patched %s ONNX session for CoreML: providers=%s",
                component_name,
                actual_providers,
            )
            patched += 1
        except Exception as exc:
            _log.warning(
                "Failed to patch %s for CoreML: %s (falling back to CPU)",
                component_name,
                exc,
            )

    if patched > 0:
        _log.info(
            "CoreML acceleration enabled for %d/%d RapidOCR ONNX sessions",
            patched,
            3,
        )
    return patched > 0


def _resolve_page_total(conv_res: ConversionResult, page_list: list[Page]) -> int:
    source = getattr(getattr(conv_res, "input", None), "file", None)
    if source:
        try:
            import pypdfium2 as _pypdfium2  # type: ignore

            return len(_pypdfium2.PdfDocument(str(source)))
        except Exception:
            pass
    try:
        doc = getattr(conv_res, "document", None)
        pages = getattr(doc, "pages", None) if doc is not None else None
        if pages is not None:
            return int(len(pages))
    except Exception:
        pass
    try:
        page_no_max = max(int(getattr(p, "page_no", 0) or 0) for p in page_list)
        return page_no_max or len(page_list)
    except Exception:
        return len(page_list)


class SafeRapidOcrModel(_RapidOcrModel):
    """Drop-in RapidOCR wrapper that copes with ``None`` OCR results.

    Docling 2.48.0 assumes ``self.reader`` always returns an object with
    ``boxes/txts/scores``. RapidOCR occasionally yields ``None`` for problematic
    crops, which crashes the extractor. We normalise the return value before the
    original list(zip(...)) call and treat anything unexpected as "no boxes".
    Remove this once Docling hardens the upstream implementation.
    """

    # NOTE: keep signature identical so StandardPdfPipeline can instantiate it.
    _rapidocr_available: Optional[bool] = None

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: RapidOcrOptions,
        accelerator_options,
    ):
        rapidocr_available = self._rapidocr_available
        if rapidocr_available is None:
            rapidocr_available = bool(
                importlib.util.find_spec("rapidocr") is not None or "rapidocr" in sys.modules
            )
            SafeRapidOcrModel._rapidocr_available = rapidocr_available

        effective_enabled = bool(enabled and rapidocr_available)
        if enabled and not rapidocr_available:
            _log.warning(
                "RapidOCR python package not found; continuing with Docling pipeline OCR disabled."
            )

        if effective_enabled:
            try:
                resolved = resolve_packaged_onnx_and_keys()

                _log.warning(
                    'SafeRapidOcrModel initial options: det=%s rec=%s cls=%s keys=%s',
                    getattr(options, 'det_model_path', None),
                    getattr(options, 'rec_model_path', None),
                    getattr(options, 'cls_model_path', None),
                    getattr(options, 'rec_keys_path', None),
                )

                if resolved.det:
                    options.det_model_path = resolved.det
                if resolved.rec:
                    options.rec_model_path = resolved.rec
                if resolved.cls:
                    options.cls_model_path = resolved.cls
                if resolved.keys:
                    options.rec_keys_path = resolved.keys

                try:
                    from rapidocr.ch_ppocr_rec import main as _rapidocr_rec_main

                    if not getattr(_rapidocr_rec_main.TextRecognizer, '_glossapi_patch', False):
                        original_get_character_dict = _rapidocr_rec_main.TextRecognizer.get_character_dict

                        def _patched_get_character_dict(self, cfg):
                            try:
                                current_keys = cfg.get('keys_path', None)
                                current_rec_keys = cfg.get('rec_keys_path', None)
                                if current_rec_keys is None and current_keys is not None:
                                    cfg['rec_keys_path'] = current_keys
                                    _log.warning('Patched RapidOCR cfg: set rec_keys_path from keys_path=%s', current_keys)
                                else:
                                    _log.warning('Patched RapidOCR cfg: existing rec_keys_path=%s keys_path=%s', current_rec_keys, current_keys)
                            except Exception:
                                _log.warning('RapidOCR cfg inspection failed', exc_info=True)
                            return original_get_character_dict(self, cfg)

                        _rapidocr_rec_main.TextRecognizer.get_character_dict = _patched_get_character_dict
                        _rapidocr_rec_main.TextRecognizer._glossapi_patch = True
                except Exception:
                    _log.warning('Failed to patch RapidOCR TextRecognizer for keys fallback', exc_info=True)

                _log.warning(
                    'SafeRapidOcrModel using packaged assets: det=%s rec=%s cls=%s keys=%s',
                    options.det_model_path,
                    options.rec_model_path,
                    options.cls_model_path,
                    options.rec_keys_path,
                )
            except Exception:
                _log.warning(
                    'SafeRapidOcrModel bootstrap failed to resolve packaged assets',
                    exc_info=True,
                )

        super().__init__(
            enabled=effective_enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )

        # -------------------------------------------------------------------
        # Override crop scale: Docling hardcodes self.scale = 3 which is
        # very expensive for full-page scanned images.  Default to 2.
        # -------------------------------------------------------------------
        if effective_enabled:
            new_scale = _get_ocr_crop_scale()
            old_scale = getattr(self, "scale", 3)
            if new_scale != old_scale:
                self.scale = new_scale
                _log.info(
                    "SafeRapidOcrModel: overrode OCR crop scale %s → %s (%.0f DPI)",
                    old_scale, new_scale, new_scale * 72,
                )

        # -------------------------------------------------------------------
        # Patch ONNX sessions for CoreML on macOS after super().__init__()
        # has created the RapidOCR reader with its ONNX sessions.
        # Only inject CoreML when the accelerator is NOT explicitly CPU.
        # -------------------------------------------------------------------
        _raw_dev = getattr(accelerator_options, "device", "")
        # Handle both plain str ("cpu") and enum (AcceleratorDevice.CPU):
        # str(AcceleratorDevice.CPU) gives 'AcceleratorDevice.CPU', not 'cpu'.
        accel_dev = (str(getattr(_raw_dev, "value", _raw_dev)) or "").lower()
        want_cpu_only = accel_dev.startswith("cpu")
        if effective_enabled and _is_macos() and hasattr(self, "reader") and not want_cpu_only:
            try:
                coreml_ok = _patch_onnx_sessions_coreml(self.reader)
                if coreml_ok:
                    _log.info("SafeRapidOcrModel: CoreML acceleration active")
                else:
                    _log.info("SafeRapidOcrModel: using CPU execution provider")
            except Exception:
                _log.warning(
                    "SafeRapidOcrModel: CoreML session patching failed",
                    exc_info=True,
                )
        elif effective_enabled and want_cpu_only:
            _log.info("SafeRapidOcrModel: CPU-only mode (CoreML injection skipped)")

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions

    def _normalise_result(self, result):
        """Return an iterable of (bbox, text, score) triples.

        RapidOCR returns ``None`` or semi-populated structures in some corner
        cases. We swallow those and log a one-line warning so the page still
        progresses through the pipeline.
        """

        if result is None:
            _log.warning("RapidOCR returned None; skipping crop")
            return []
        boxes = getattr(result, "boxes", None)
        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or txts is None or scores is None:
            _log.warning("RapidOCR returned incomplete data; treating crop as empty")
            return []
        try:
            return list(zip(boxes.tolist(), txts, scores))
        except Exception as exc:  # pragma: no cover - defensive only
            _log.warning("RapidOCR result normalisation failed: %s", exc)
            return []

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        page_list = list(page_batch)
        progress = None
        progress_env = os.getenv("GLOSSAPI_OCR_PAGE_PROGRESS", "1").strip().lower()
        if progress_env not in {"0", "false", "no"}:
            total_pages = getattr(conv_res, "_glossapi_page_total", None)
            if total_pages is None:
                total_pages = _resolve_page_total(conv_res, page_list)
                setattr(conv_res, "_glossapi_page_total", total_pages)

            progress = getattr(conv_res, "_glossapi_ocr_progress", None)
            if progress is None:
                source = getattr(getattr(conv_res, "input", None), "file", None)
                label = Path(source).name if source else "OCR"
                try:
                    progress = tqdm(
                        total=total_pages,
                        desc=f"OCR {label}",
                        unit="page",
                        disable=not sys.stderr.isatty(),
                    )
                except Exception:
                    progress = None
                setattr(conv_res, "_glossapi_ocr_progress", progress)
                setattr(conv_res, "_glossapi_ocr_done", 0)

        page_timeout = _get_page_timeout()
        # Reuse a single ThreadPoolExecutor for the entire batch to avoid
        # the overhead of creating/destroying one per OCR rectangle.
        _pool = ThreadPoolExecutor(max_workers=1)

        try:
            for page in page_list:
                assert page._backend is not None
                if not page._backend.is_valid():
                    yield page
                    continue

                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells: list = []
                    page_no = getattr(page, "page_no", "?")
                    timed_out = False

                    for ocr_rect in ocr_rects:
                        if timed_out:
                            break
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        im = numpy.array(high_res_image)

                        # ---------- per-rect OCR with timeout ----------
                        raw_result = None
                        t0 = time.monotonic()
                        try:
                            future = _pool.submit(
                                self.reader,
                                im,
                                use_det=self.options.use_det,
                                use_cls=self.options.use_cls,
                                use_rec=self.options.use_rec,
                            )
                            raw_result = future.result(timeout=page_timeout)
                        except FuturesTimeoutError:
                            elapsed = time.monotonic() - t0
                            _log.warning(
                                "OCR timed out after %.1fs on page %s "
                                "(image %dx%d); skipping remaining rects",
                                elapsed,
                                page_no,
                                im.shape[1] if im is not None else 0,
                                im.shape[0] if im is not None else 0,
                            )
                            timed_out = True
                            del high_res_image
                            del im
                            continue
                        except Exception as exc:
                            _log.warning(
                                "OCR error on page %s rect: %s", page_no, exc
                            )
                            del high_res_image
                            del im
                            continue

                        elapsed = time.monotonic() - t0
                        if elapsed > 5.0:
                            _log.info(
                                "OCR page %s rect took %.1fs (image %dx%d, scale=%.1f)",
                                page_no,
                                elapsed,
                                im.shape[1],
                                im.shape[0],
                                self.scale,
                            )
                        # ---------- end timeout wrapper ----------

                        result = self._normalise_result(raw_result)
                        del high_res_image
                        del im

                        if not result:
                            continue

                        cells = [
                            TextCell(
                                index=ix,
                                text=line[1],
                                orig=line[1],
                                confidence=line[2],
                                from_ocr=True,
                                rect=BoundingRectangle.from_bounding_box(
                                    BoundingBox.from_tuple(
                                        coord=(
                                            (line[0][0][0] / self.scale) + ocr_rect.l,
                                            (line[0][0][1] / self.scale) + ocr_rect.t,
                                            (line[0][2][0] / self.scale) + ocr_rect.l,
                                            (line[0][2][1] / self.scale) + ocr_rect.t,
                                        ),
                                        origin=CoordOrigin.TOPLEFT,
                                    )
                                ),
                            )
                            for ix, line in enumerate(result)
                        ]
                        all_ocr_cells.extend(cells)

                    self.post_process_cells(all_ocr_cells, page)

                from docling.datamodel.settings import settings

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                if progress is not None:
                    try:
                        progress.update(1)
                        done = int(getattr(conv_res, "_glossapi_ocr_done", 0)) + 1
                        setattr(conv_res, "_glossapi_ocr_done", done)
                        total_pages = int(getattr(conv_res, "_glossapi_page_total", 0) or 0)
                        if total_pages > 0 and done >= total_pages:
                            progress.close()
                            setattr(conv_res, "_glossapi_ocr_progress", None)
                    except Exception:
                        pass
                yield page
        finally:
            # Shut down the thread pool used for timeout-guarded OCR.
            try:
                _pool.shutdown(wait=False)
            except Exception:
                pass
            # Only close the progress bar if ALL document pages have been
            # processed.  Docling sends pages in batches, so __call__ is
            # invoked multiple times for the same conv_res.  Closing here
            # would reset the bar to 0 at the start of the next batch.
            if progress is not None:
                try:
                    done = int(getattr(conv_res, "_glossapi_ocr_done", 0))
                    total = int(getattr(conv_res, "_glossapi_page_total", 0) or 0)
                    if total > 0 and done >= total:
                        progress.close()
                        setattr(conv_res, "_glossapi_ocr_progress", None)
                    # Otherwise keep the bar alive for the next batch.
                except Exception:
                    pass


def patch_docling_rapidocr() -> bool:
    """Replace Docling's RapidOcrModel with the safe shim if available."""

    try:
        import docling.models.rapid_ocr_model as rapid_module
    except Exception:  # pragma: no cover - Docling missing
        return False

    current = getattr(rapid_module, "RapidOcrModel", None)
    if current is SafeRapidOcrModel:
        return False

    rapid_module.RapidOcrModel = SafeRapidOcrModel
    try:
        from docling.models.factories import get_ocr_factory  # type: ignore
        import logging
    except Exception:
        return True

    try:
        factory = get_ocr_factory()
        options_type = SafeRapidOcrModel.get_options_type()

        if hasattr(factory, "classes"):
            factory.classes[options_type] = SafeRapidOcrModel
        elif hasattr(factory, "_classes"):
            factory._classes[options_type] = SafeRapidOcrModel
        logging.getLogger(__name__).info(
            "Registered SafeRapidOcrModel for %s", options_type
        )
        try:
            from docling.pipeline import standard_pdf_pipeline as _std_pdf  # type: ignore
            from docling.datamodel.pipeline_options import RapidOcrOptions  # type: ignore
            from functools import lru_cache
        except Exception as _exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).warning(
                "Docling factory patch limited to local mutation: %s", _exc
            )
        else:
            original_get_factory = getattr(
                _std_pdf.get_ocr_factory, "__wrapped__", _std_pdf.get_ocr_factory
            )

            def _ensure_safe(factory_obj):
                try:
                    current = factory_obj.classes.get(RapidOcrOptions)
                    if current is not SafeRapidOcrModel:
                        factory_obj.classes[RapidOcrOptions] = SafeRapidOcrModel
                except AttributeError:
                    current = getattr(factory_obj, "_classes", {}).get(RapidOcrOptions)
                    if current is not SafeRapidOcrModel:
                        getattr(factory_obj, "_classes", {})[RapidOcrOptions] = SafeRapidOcrModel
                return factory_obj

            @lru_cache(maxsize=None)
            def _patched_get_ocr_factory(allow_external_plugins: bool = False):
                return _ensure_safe(original_get_factory(allow_external_plugins))

            _patched_get_ocr_factory.__wrapped__ = original_get_factory  # type: ignore[attr-defined]
            _std_pdf.get_ocr_factory = _patched_get_ocr_factory  # type: ignore[attr-defined]
            try:
                _ensure_safe(_std_pdf.get_ocr_factory(False))
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).warning(
            "Failed to re-register SafeRapidOcrModel: %s", exc
        )
    return True


__all__ = ["SafeRapidOcrModel", "patch_docling_rapidocr"]
