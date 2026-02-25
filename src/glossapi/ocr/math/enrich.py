from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore

from docling_core.types.doc import DocItemLabel  # type: ignore


@dataclass
class RoiEntry:
    page_no: int  # 1-based
    bbox: Any     # Docling BoundingBox
    label: str
    per_page_ix: int


class PageRasterCache:
    """Thread-safe page rasterizer with a true LRU eviction policy.

    Improvements vs the previous implementation:

    * Uses ``collections.OrderedDict`` for O(1) LRU moves instead of a plain
      ``dict`` with FIFO eviction (``next(iter())`` gave insertion-order, not
      recency order, so hot pages with many formulas were evicted first).
    * Default capacity raised from 4 → 16 pages.  On M-series hardware with
      unified memory, 16 × ~30 MB (A3 page at 220 DPI) ≈ 480 MB — well within
      budget for a typical 8–16 GB Mac.
    * Thread-safe: a ``threading.Lock`` guards all cache mutations so the
      background prefetch thread and the main inference loop never race.
    * Exposes ``page_height_pts()`` to read the native page height in PDF
      points without rasterizing, enabling single-render per ROI (see below).
    """

    def __init__(self, pdf_path: Path, max_cached_pages: int = 16):
        if pdfium is None:
            raise RuntimeError("pypdfium2 not available; cannot rasterize PDF")
        self.doc = pdfium.PdfDocument(str(pdf_path))
        # OrderedDict preserves insertion order; we bump an entry to the end on
        # each access so the front is always the least-recently-used entry.
        self._cache: OrderedDict[Tuple[int, int], Any] = OrderedDict()
        self.max_cached = int(max_cached_pages)
        self._lock = threading.Lock()

    def page_height_pts(self, page_no_1b: int) -> float:
        """Return the page height in PDF points (1 pt = 1/72 inch).

        This is a metadata read — pypdfium2 reads only the page dictionary,
        not the content stream, so it is effectively free.
        """
        page = self.doc[int(page_no_1b) - 1]
        # pypdfium2 exposes width/height via get_width() / get_height()
        return float(page.get_height())

    def get_pil(self, page_no_1b: int, dpi: int):
        """Return a PIL Image of the page at the requested DPI.

        On a cache hit the entry is promoted to the MRU end.  On a miss the
        page is rasterised, cached, and the LRU entry is evicted if the cache
        is full.  All mutations are protected by ``self._lock`` so the method
        is safe to call from multiple threads simultaneously.
        """
        key = (int(page_no_1b), int(dpi))
        with self._lock:
            if key in self._cache:
                # Promote to MRU end — O(1) with OrderedDict
                self._cache.move_to_end(key)
                return self._cache[key]

        # Rasterize outside the lock: pypdfium2 rendering is CPU-bound and can
        # run concurrently with other cache readers on different pages.
        page = self.doc[int(page_no_1b) - 1]
        scale = float(dpi) / 72.0
        bm = page.render(scale=scale)
        im = bm.to_pil()

        with self._lock:
            # Another thread may have rasterized the same page concurrently;
            # accept whichever result arrived first and discard the duplicate.
            if key not in self._cache:
                self._cache[key] = im
                # Evict LRU entries until we are within capacity.
                while len(self._cache) > self.max_cached:
                    self._cache.popitem(last=False)  # remove LRU (front)
        return self._cache[key]


class _PagePrefetcher:
    """Background thread that pre-rasterizes upcoming pages into PageRasterCache.

    This implements a simple producer-consumer pattern: the main thread is the
    consumer (crops → batch → GPU inference) and the prefetcher is the producer
    (pypdfium2 rasterization, CPU-bound).

    On Apple Silicon the CPU cores and the ANE/GPU share unified memory but run
    independently, so overlapping CPU rasterization with ANE inference eliminates
    most of the GPU stall time between consecutive batches.
    """

    def __init__(self, rc: PageRasterCache, rois: Sequence[RoiEntry], dpi_for_roi_fn):
        self._rc = rc
        self._rois = rois
        self._dpi_for_roi_fn = dpi_for_roi_fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="enrich-prefetch")
        self._thread.start()

    def _run(self) -> None:
        for roi in self._rois:
            if self._stop.is_set():
                break
            try:
                dpi = self._dpi_for_roi_fn(roi)
                self._rc.get_pil(roi.page_no, dpi)
            except Exception:
                pass  # main thread will surface the error when it tries the same page

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _dpi_for_bbox(px_h: float, *, base: int = 220, lo: int = 180, hi: int = 320) -> int:
    # Simple heuristic: small crops get higher dpi, big crops get lower
    if px_h < 40:
        return int(min(hi, max(base + 60, lo)))
    if px_h < 120:
        return int(min(hi, max(base + 20, lo)))
    if px_h > 800:
        return int(max(lo, base - 40))
    return int(base)


def _crop_box_pixels(bbox, pil_h: int, dpi: int) -> Tuple[int, int, int, int]:
    # Convert Docling bbox to top-left pixel coords using the page pixel height
    try:
        b2 = bbox.scaled(scale=float(dpi) / 72.0).to_top_left_origin(page_height=float(pil_h))
        l, t, r, b = map(int, (b2.l, b2.t, b2.r, b2.b))
        return max(0, l), max(0, t), max(0, r), max(0, b)
    except Exception:
        # fallback: assume bbox already pixel with top-left origin
        l, t, r, b = map(int, (getattr(bbox, 'l', 0), getattr(bbox, 't', 0), getattr(bbox, 'r', 0), getattr(bbox, 'b', 0)))
        return max(0, l), max(0, t), max(0, r), max(0, b)


def enrich_from_docling_json(
    json_path: Path,
    pdf_path: Path,
    out_md_path: Path,
    out_map_jsonl: Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    pad_px: int = 3,
    dpi_base: int = 220,
    dpi_lo: int = 180,
    dpi_hi: int = 320,
    targets: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Load a DoclingDocument JSON and enrich FORMULA/CODE items using Docling CodeFormula.

    Writes final Markdown and a latex_map.jsonl sidecar; returns a small metrics dict.
    """
    from glossapi.ocr.utils.json_io import load_docling_json  # type: ignore
    from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions  # type: ignore
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice  # type: ignore
    from docling.datamodel.base_models import ItemAndImageEnrichmentElement  # type: ignore

    doc = load_docling_json(json_path)
    rc = PageRasterCache(pdf_path, max_cached_pages=16)

    # Collect ROIs (optionally filter by explicit (page_no, per_page_ix) targets)
    rois: list[RoiEntry] = []
    per_page_ix: dict[int, int] = {}
    it = getattr(doc, 'iterate_items', None)
    if not callable(it):
        raise RuntimeError("DoclingDocument missing iterate_items()")
    targets_set = set((int(p), int(ix)) for (p, ix) in (targets or []))
    for element, _level in it():
        lab = str(getattr(element, 'label', '')).lower()
        if lab not in {"formula", "code"}:
            continue
        prov = getattr(element, 'prov', []) or []
        if not prov:
            continue
        p = prov[0]
        page_no = int(getattr(p, 'page_no', 0))
        bbox = getattr(p, 'bbox', None)
        if not page_no or bbox is None:
            continue
        per_page_ix[page_no] = per_page_ix.get(page_no, 0) + 1
        cur_ix = per_page_ix[page_no]
        if targets_set and (page_no, cur_ix) not in targets_set:
            continue
        rois.append(RoiEntry(page_no=page_no, bbox=bbox, label=lab, per_page_ix=cur_ix))

    if not rois:
        # Nothing to enrich; still emit the base markdown (and fall back to raw text if empty)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_markdown(out_md_path)
        try:
            rendered = out_md_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            rendered = ""
        if not rendered.strip():
            # Some Docling docs store text only in the lightweight texts list; surface it so the
            # markdown is not empty (important for tests and downstream auditability).
            fallback: list[str] = []
            text_items = getattr(doc, "texts", None)
            if text_items:
                for item in text_items:
                    value = getattr(item, "text", None) if hasattr(item, "text") else None
                    if value:
                        fallback.append(str(value))
            if fallback:
                out_md_path.write_text("\n\n".join(fallback), encoding="utf-8")
        return {"items": 0, "accepted": 0, "time_sec": 0.0}

    # Map string device name to AcceleratorDevice enum; passing a raw string causes Docling
    # to fail its own device validation and fall back to CPU silently.
    _DEVICE_MAP = {
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "cpu": AcceleratorDevice.CPU,
        "auto": AcceleratorDevice.AUTO,
    }
    _device_enum = _DEVICE_MAP.get(str(device or "").strip().lower(), AcceleratorDevice.AUTO)
    acc = AcceleratorOptions(device=_device_enum)
    opts = CodeFormulaModelOptions(do_code_enrichment=True, do_formula_enrichment=True)
    model: Optional[CodeFormulaModel] = None
    try:
        model = CodeFormulaModel(enabled=True, artifacts_path=None, options=opts, accelerator_options=acc)
        try:
            from .earlystop import attach_early_stop
            attach_early_stop(model)
        except Exception:
            pass
        try:
            setattr(model, "elements_batch_size", int(batch_size))
        except Exception:
            pass

        t0 = time.time()
        accepted = 0
        written = 0
        out_map_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_map_jsonl.open("w", encoding="utf-8") as fp:
            fp.write("")

        batch: list[ItemAndImageEnrichmentElement] = []
        binfo: list[Tuple[int, int, int]] = []

        from glossapi.text_sanitize import accept_latex, load_latex_policy, sanitize_latex, tail_run  # type: ignore

        def _sanitize_latex(text: str, *, max_len: int = 3000, max_tail_repeats: int = 50) -> tuple[str, dict]:
            s = text or ""
            info = {
                "orig_len": len(s),
                "truncated_by_repeat": False,
                "truncated_by_len": False,
                "tail_token": "",
                "tail_run": 0,
            }
            toks = s.split()
            if toks:
                last = toks[-1]
                run = 1
                i = len(toks) - 2
                while i >= 0 and toks[i] == last:
                    run += 1
                    i -= 1
                info["tail_token"] = last
                info["tail_run"] = run
                if run > int(max_tail_repeats):
                    keep = int(max_tail_repeats)
                    new_toks = toks[: (i + 1 + keep)]
                    s = " ".join(new_toks)
                    info["truncated_by_repeat"] = True
            if len(s) > int(max_len):
                cut_ws = max(
                    s.rfind(" ", 0, int(max_len)),
                    s.rfind("\n", 0, int(max_len)),
                    s.rfind("\t", 0, int(max_len)),
                )
                cut = cut_ws if cut_ws != -1 else int(max_len)
                s = s[:cut].rstrip()
                info["truncated_by_len"] = True
            return s, info

        def _env_int(name: str, default: int) -> int:
            try:
                v = os.getenv(name)
                return int(v) if v is not None and str(v).strip() != "" else default
            except Exception:
                return default

        policy = load_latex_policy()

        def _tail_run(s: str) -> int:
            toks = s.split()
            if not toks:
                return 0
            last = toks[-1]
            run = 1
            i = len(toks) - 2
            while i >= 0 and toks[i] == last:
                run += 1
                i -= 1
            return run

        def flush_batch() -> None:
            nonlocal accepted, batch, binfo, written
            if not batch:
                return
            t_call = time.time()
            for _out_item in model(doc, batch):  # type: ignore[misc]
                pass
            dt_ms = int((time.time() - t_call) * 1000.0)
            per_ms = int(dt_ms / max(1, len(batch)))
            out_lines: list[str] = []
            _engine = "code_formula"
            try:
                import docling as _dl  # type: ignore

                _engine_ver = getattr(_dl, "__version__", "") or ""
            except Exception:
                _engine_ver = ""
            for (page_no, ix, dpi_used), el in zip(binfo, batch):
                latex = getattr(el.item, "text", "") or ""
                do_post = True
                if policy.post_only_failed:
                    tr = 0
                    try:
                        tr = tail_run(latex)
                    except Exception:
                        tr = 0
                    do_post = (tr > policy.post_repeat_gate) or (len(latex) > policy.post_max_chars)
                sinfo = {
                    "orig_len": len(latex),
                    "truncated_by_repeat": False,
                    "truncated_by_len": False,
                    "tail_token": "",
                    "tail_run": 0,
                }
                if do_post:
                    sanitized, sinfo = sanitize_latex(latex, policy)
                    if sanitized != latex:
                        try:
                            setattr(el.item, "text", sanitized)
                        except Exception:
                            pass
                    latex = sanitized
                ok = accept_latex(latex) >= 1.0
                if ok:
                    accepted += 1
                row = {
                    "page_no": int(page_no),
                    "item_index": int(ix),
                    "latex": latex,
                    "accept_score": 1.0 if ok else 0.0,
                    "compile_ok": False,
                    "dpi": int(dpi_used),
                    "orig_len": int(sinfo.get("orig_len", len(latex))),
                    "truncated_by_repeat": bool(sinfo.get("truncated_by_repeat", False)),
                    "truncated_by_len": bool(sinfo.get("truncated_by_len", False)),
                    "tail_token": str(sinfo.get("tail_token", "")),
                    "tail_run": int(sinfo.get("tail_run", 0)),
                    "post_applied": bool(do_post),
                    "engine": _engine,
                    "engine_version": _engine_ver,
                    "time_ms": int(per_ms),
                }
                out_lines.append(json.dumps(row, ensure_ascii=False))
            if out_lines:
                with out_map_jsonl.open("a", encoding="utf-8") as fp:
                    fp.write("\n".join(out_lines) + "\n")
            written += len(out_lines)
            if written % max(1, batch_size) == 0:
                print(f"[Phase-2] Wrote {written}/{len(rois)} items for current doc…")
            batch = []
            binfo = []

        # -----------------------------------------------------------------------
        # Helper: compute the adaptive DPI for an ROI *without* rasterizing the
        # page first.  We estimate the crop height from the native Docling bbox
        # coordinates (which are in PDF points, 1 pt = 1/72 in) and the
        # page's metadata height — a free pypdfium2 dict read, no pixel data.
        # This eliminates the previous double-rasterization: the old code called
        # rc.get_pil(page_no, base_dpi) solely to obtain im.height, then called
        # rc.get_pil(page_no, adaptive_dpi) again — two cache entries per ROI.
        # -----------------------------------------------------------------------
        def _adaptive_dpi_for_roi(entry: RoiEntry) -> int:
            """Return the rasterization DPI for an ROI using native PDF coordinates.

            Avoids a full page render just to determine the crop height.
            """
            try:
                # Native bbox height in points (72 DPI)
                bbox_h_pts = abs(
                    float(getattr(entry.bbox, 'b', 0)) -
                    float(getattr(entry.bbox, 't', 0))
                )
                # Estimate the crop height in pixels at base_dpi
                est_px_h = bbox_h_pts * (float(dpi_base) / 72.0)
                return _dpi_for_bbox(est_px_h, base=dpi_base, lo=dpi_lo, hi=dpi_hi)
            except Exception:
                return int(dpi_base)

        # -----------------------------------------------------------------------
        # Producer: pre-rasterize pages for upcoming ROIs in a background thread
        # while the main thread processes the current batch through the model.
        # On Apple Silicon this overlaps CPU (pypdfium2) with ANE/GPU inference,
        # keeping both execution units busy continuously.
        # -----------------------------------------------------------------------
        _prefetcher = _PagePrefetcher(rc, rois, _adaptive_dpi_for_roi)
        try:
            # Inner try so we always stop the prefetcher, even on exceptions.

            print(f"[Phase-2] {json_path.stem}: {len(rois)} items to enrich …")
            for entry in rois:
                # Single render: compute the adaptive DPI from native coordinates
                # (no extra page render needed to measure im.height).
                dpi = _adaptive_dpi_for_roi(entry)
                im = rc.get_pil(entry.page_no, dpi)  # likely already in cache
                l, t, r, b = _crop_box_pixels(entry.bbox, pil_h=im.height, dpi=dpi)
                l = max(0, l - int(pad_px))
                t = max(0, t - int(pad_px))
                r = min(im.width, r + int(pad_px))
                b = min(im.height, b + int(pad_px))
                crop = im.crop((l, t, r, b))
                _item = getattr(entry, "item", None) or _find_item(doc, entry)
                if _item is None:
                    # No matching code/formula element found in the document; skip this ROI
                    # rather than passing a wrong-label item to CodeFormulaModel.
                    continue
                batch.append(
                    ItemAndImageEnrichmentElement(
                        item=_item,
                        image=crop,
                    )
                )
                binfo.append((entry.page_no, entry.per_page_ix, int(dpi)))
                if len(batch) >= int(batch_size):
                    flush_batch()
            flush_batch()

            out_md_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save_as_markdown(out_md_path)

            return {"items": len(rois), "accepted": accepted, "time_sec": time.time() - t0}
        finally:
            _prefetcher.stop()
    finally:
        try:
            if model is not None:
                try:
                    inner = getattr(model, "_model", None)
                    if inner is not None:
                        inner.to("cpu")
                except Exception:
                    pass
                try:
                    model._processor = None  # type: ignore[attr-defined]
                except Exception:
                    pass
            import torch  # type: ignore

            # Release CUDA memory (Linux/Windows)
            cuda_iface = getattr(torch, "cuda", None)
            if cuda_iface and cuda_iface.is_available():  # type: ignore[attr-defined]
                try:
                    cuda_iface.empty_cache()
                except Exception:
                    pass
                try:
                    ipc_collect = getattr(cuda_iface, "ipc_collect", None)
                    if callable(ipc_collect):
                        ipc_collect()
                except Exception:
                    pass

            # Release Metal / Apple Silicon unified memory buffers.
            # torch.mps.empty_cache() signals Metal to compact its allocator
            # heap so fragmented GPU buffers are returned to the OS pool before
            # the next document is processed — especially important when running
            # many documents sequentially with limited swap.
            mps_iface = getattr(torch, "mps", None)
            if mps_iface is not None:
                try:
                    synchronize = getattr(mps_iface, "synchronize", None)
                    if callable(synchronize):
                        synchronize()  # wait for all Metal kernels to finish
                except Exception:
                    pass
                try:
                    empty_cache = getattr(mps_iface, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()
                except Exception:
                    pass
        except Exception:
            pass


def _find_item(doc, entry: RoiEntry):
    # Find the actual NodeItem instance matching page_no + per_page_ix for FORMULA/CODE.
    # Returns None if no matching element is found (caller must guard).
    c = 0
    last_seen = None
    for element, _level in doc.iterate_items():  # type: ignore[attr-defined]
        lab = str(getattr(element, 'label', '')).lower()
        if lab != entry.label:
            continue
        prov = getattr(element, 'prov', []) or []
        if not prov:
            continue
        p = prov[0]
        pn = int(getattr(p, 'page_no', 0))
        if pn != entry.page_no:
            continue
        last_seen = element
        c += 1
        if c == entry.per_page_ix:
            return element
    # Fallback: last element with the correct label on the correct page (not the absolute
    # last iterated element, which could be any label and would cause Docling to raise
    # "Label must be either code or formula").
    return last_seen
