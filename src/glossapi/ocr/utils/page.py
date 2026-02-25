from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None


def _page_count(pdf_path: Path) -> int:
    """Return the number of pages in *pdf_path*, or 0 if unparseable."""
    if _pypdfium2 is None:
        return 0
    try:
        _doc = _pypdfium2.PdfDocument(str(pdf_path))
        try:
            return len(_doc)
        finally:
            _doc.close()
    except Exception:
        return 0



def is_mostly_blank_pix(pixmap, *, tolerance: int = 8, max_fraction: float = 0.0015) -> bool:
    """Heuristic check whether a PyMuPDF pixmap is mostly blank.

    Accepts a `fitz.Pixmap` instance. Returns True if the page is blank-ish.
    """
    buf = getattr(pixmap, "samples", None)
    if not buf:
        return True
    channels = 4 if getattr(pixmap, "alpha", False) else 3
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return True
    arr = arr.reshape(-1, channels)
    if channels == 4:
        arr = arr[:, :3]
    if arr.size == 0:
        return True
    if arr.shape[0] > 65536:
        samples = arr[::64]
    else:
        samples = arr
    samples16 = samples.astype(np.int16, copy=False)
    base = samples16[0]
    diff = np.abs(samples16 - base)
    if diff.max() <= tolerance:
        return True
    mask = np.any(diff > tolerance, axis=1)
    return float(mask.mean()) <= max_fraction

