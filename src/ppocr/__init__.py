"""Compatibility shim for PaddleOCR's `ppocr` package.

Some integrations import `ppocr` as a top-level module. The PyPI
`paddleocr` package bundles it under its own package directory, but
does not expose it as a top-level import. We load the bundled package
directly to satisfy `ppocr.*` imports without importing `paddleocr`.
"""
from __future__ import annotations

from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
import sys

spec = find_spec("paddleocr")
if spec is None or not spec.submodule_search_locations:
	raise ModuleNotFoundError("paddleocr is not installed; cannot expose ppocr")

base = Path(next(iter(spec.submodule_search_locations)))
ppocr_dir = base / "ppocr"
ppocr_init = ppocr_dir / "__init__.py"
if not ppocr_init.exists():
	__path__ = [str(ppocr_dir)]
	__package__ = __name__
	__spec__ = spec_from_file_location(
		__name__,
		ppocr_init,
		submodule_search_locations=[str(ppocr_dir)],
	)
else:
	ppocr_spec = spec_from_file_location(
		__name__,
		ppocr_init,
		submodule_search_locations=[str(ppocr_dir)],
	)
	if ppocr_spec is None or ppocr_spec.loader is None:
		raise ModuleNotFoundError("Unable to load paddleocr.ppocr package")

	module = module_from_spec(ppocr_spec)
	sys.modules[__name__] = module
	ppocr_spec.loader.exec_module(module)
