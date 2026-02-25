"""Shared CLI prompt utilities and run configuration for GlossAPI wizards.

This module centralises the raw gum subprocess machinery, ANSI-strip helper,
and simple TTY fallbacks so that ``setup_wizard`` and ``pipeline_wizard`` do
not duplicate them.  It also hosts ``RunConfig`` — the single typed descriptor
that replaces the three scattered ``*_kwargs`` dicts previously passed between
the wizard and the pipeline runner.

Each wizard retains its own ``_require_gum()`` guard and ``_run_gum()``
wrapper (because their TTY/fallback policies intentionally differ) and binds
them to the shared ``_gum_*_impl`` helpers via 1-line thin wrappers.

Public API
----------
RunConfig             — typed pipeline run descriptor (dataclass).
_ANSI_RE              — compiled regex for stripping ANSI escape codes.
_run_gum_raw          — low-level: invoke a gum sub-command; return (rc, output).
_open_tty             — open /dev/tty for non-gum fallback I/O.
_simple_select        — numbered list selection for non-TTY / SIMPLE mode.
_simple_confirm       — y/n prompt for non-TTY / SIMPLE mode.
_simple_text          — free-text prompt for non-TTY / SIMPLE mode.
GumRunFn              — type alias for the gum runner callable.
_gum_choose_impl      — shared gum choose logic; always returns List[str].
_gum_confirm_impl     — shared gum confirm logic.
_gum_input_impl       — shared gum input logic.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _run_gum_raw(args: list[str]) -> tuple[int, str]:
    """Execute a gum sub-command, returning ``(returncode, stripped_output)``.

    Opens ``/dev/tty`` directly so gum's interactive UI is rendered even when
    stdout is piped (e.g. when the caller captures output for other purposes).
    Falls back to ``capture_output=True`` when ``/dev/tty`` cannot be opened,
    which happens in most CI/non-interactive environments.

    Callers are responsible for running any ``_require_gum()`` guard before
    calling this function so that missing-binary errors surface cleanly.
    """
    try:
        tty_in = open("/dev/tty", "rb")
        tty_out = open("/dev/tty", "wb")
    except Exception:
        proc = subprocess.run(["gum", *args], text=True, capture_output=True)
        output = _ANSI_RE.sub("", proc.stdout or "")
        return proc.returncode, output.strip()

    proc = subprocess.run(
        ["gum", *args],
        stdin=tty_in,
        stdout=subprocess.PIPE,
        stderr=tty_out,
        text=True,
    )
    tty_in.close()
    tty_out.close()
    output = _ANSI_RE.sub("", proc.stdout or "")
    return proc.returncode, output.strip()


def _open_tty() -> Optional[object]:
    """Open ``/dev/tty`` for read/write; return ``None`` if unavailable."""
    try:
        return open("/dev/tty", "r+")
    except Exception:
        return None


def _simple_select(
    label: str,
    choices: list[str],
    default: str,
    tty: Optional[object],
) -> str:
    """Numbered-list selection for non-gum / non-interactive mode.

    Parameters
    ----------
    label:
        Prompt header text.
    choices:
        List of option strings.
    default:
        The option that is chosen when the user presses Enter.
    tty:
        Open file object for direct tty I/O, or ``None`` to use stdin/stdout.
    """
    default_idx = choices.index(default) + 1 if default in choices else 1

    def _write(msg: str) -> None:
        if tty:
            tty.write(msg)  # type: ignore[union-attr]
            tty.flush()  # type: ignore[union-attr]
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def _readline() -> str:
        if tty:
            return tty.readline()  # type: ignore[union-attr]
        return input()

    _write(f"{label}:\n")
    for idx, choice in enumerate(choices, start=1):
        _write(f"  {idx}) {choice}\n")
    while True:
        _write(f"Select [default {default_idx}]: ")
        answer = _readline().strip()
        if not answer:
            return choices[default_idx - 1]
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        _write("Invalid choice. Try again.\n")


def _simple_confirm(label: str, default: bool, tty: Optional[object]) -> bool:
    """y/n prompt for non-gum / non-interactive mode."""
    default_char = "Y" if default else "n"
    while True:
        prompt = f"{label} [Y/n] (default {default_char}): "
        if tty:
            tty.write(prompt)  # type: ignore[union-attr]
            tty.flush()  # type: ignore[union-attr]
            answer = tty.readline().strip().lower()  # type: ignore[union-attr]
        else:
            answer = input(prompt).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        msg = "Please answer y or n.\n"
        if tty:
            tty.write(msg)  # type: ignore[union-attr]
            tty.flush()  # type: ignore[union-attr]
        else:
            print(msg, end="")


def _simple_text(label: str, default: str, tty: Optional[object]) -> str:
    """Free-text input for non-gum / non-interactive mode."""
    prompt = f"{label} [default: {default}]: "
    if tty:
        tty.write(prompt)  # type: ignore[union-attr]
        tty.flush()  # type: ignore[union-attr]
        answer = tty.readline().strip()  # type: ignore[union-attr]
    else:
        answer = input(prompt).strip()
    return answer or default


# ---------------------------------------------------------------------------
# RunConfig — typed pipeline run descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunConfig:
    """Fully-typed description of a single GlossAPI pipeline run.

    Built interactively by ``_run_wizard`` from user answers and passed
    verbatim to ``_run_pipeline``.  The three opaque ``*_kwargs`` dicts that
    existed previously are replaced by explicit typed fields; ``ocr_kwargs``,
    ``extract_kwargs``, and ``clean_kwargs`` are read-only properties that
    materialise the dicts expected by the ``Corpus`` phase methods.

    Mutability is intentional: the internal fallback logic in ``_run_pipeline``
    (e.g. MinerU → RapidOCR) writes back to ``backend`` so that subsequent
    reads of ``ocr_kwargs`` automatically reflect the change.
    """

    # ---- core ---------------------------------------------------------------
    input_dir: Path
    output_dir: Path
    input_format: str            # "pdf" | "md"
    phases: List[str]
    confirm_each: bool = False
    log_level: int = dataclasses.field(default_factory=lambda: logging.INFO)

    # ---- backend ------------------------------------------------------------
    backend: Optional[str] = None          # "mineru" | "rapidocr" | …, or None
    device: Optional[str] = None           # "cpu" | "cuda" | "mps" | None
    math_enhance: bool = True
    ocr_mode: str = "ocr_bad_then_math"
    fix_bad: bool = True
    mineru_backend: Optional[str] = None   # "hybrid-auto-engine" | "pipeline"

    # ---- extract ------------------------------------------------------------
    phase1_backend: Optional[str] = None   # "safe" | "docling" | None
    accel_type: Optional[str] = None       # "CUDA" | "MPS" | "CPU" | None

    # ---- clean --------------------------------------------------------------
    drop_bad: bool = True

    # ---- materialised kwarg dicts -------------------------------------------

    @property
    def ocr_kwargs(self) -> Optional[dict]:
        """Return the kwarg dict for ``Corpus.ocr()``, or ``None`` when no OCR backend is set."""
        if self.backend is None:
            return None
        kw: dict = {
            "backend": self.backend,
            "math_enhance": self.math_enhance,
            "mode": self.ocr_mode,
            "fix_bad": self.fix_bad,
        }
        if self.device is not None:
            kw["device"] = self.device
        if self.mineru_backend is not None:
            kw["mineru_backend"] = self.mineru_backend
        return kw

    @property
    def extract_kwargs(self) -> Optional[dict]:
        """Return the kwarg dict for ``Corpus.extract()``, or ``None`` when empty."""
        kw: dict = {}
        if self.phase1_backend is not None:
            kw["phase1_backend"] = self.phase1_backend
        if self.accel_type is not None:
            kw["accel_type"] = self.accel_type
        return kw or None

    @property
    def clean_kwargs(self) -> dict:
        """Return the kwarg dict for ``Corpus.clean()``."""
        return {"drop_bad": self.drop_bad}


# ---------------------------------------------------------------------------
# Shared gum prompt implementations
# ---------------------------------------------------------------------------

#: Type alias for the gum runner callable each wizard supplies.
GumRunFn = Callable[[List[str]], tuple[int, str]]


def _gum_choose_impl(
    run_fn: GumRunFn,
    label: str,
    choices: List[str],
    *,
    default: Optional[Any] = None,
    multi: bool = False,
) -> List[str]:
    """Shared gum-choose logic; always returns a list of selected strings.

    Returns ``[]`` on user cancel (non-zero gum exit code) so callers can
    decide whether to abort (single-select wrappers typically raise
    ``typer.Exit``) or to treat empty as a valid answer (multi-select).

    Parameters
    ----------
    run_fn:
        The wizard's local ``_run_gum`` callable, which applies the correct
        ``_require_gum()`` guard before delegating to ``_run_gum_raw``.
    label:
        Header text shown above the list.
    choices:
        Items to display.  For multi-select, the returned list contains only
        the chosen items.
    default:
        Pre-selected item(s).  Pass a ``str`` for single-select, a ``list``
        of strings for multi-select.
    multi:
        When ``True`` gum is invoked with ``--no-limit`` so the user can
        select multiple items.
    """
    args = ["choose", "--header", label]
    if multi:
        args.append("--no-limit")
    if default:
        if isinstance(default, list):
            for item in default:
                args.extend(["--selected", item])
        else:
            args.extend(["--selected", default])
    args.extend(choices)
    code, output = run_fn(args)
    if code != 0 or not output:
        return []
    if multi:
        return [line for line in output.splitlines() if line.strip()]
    return [output]


def _gum_confirm_impl(run_fn: GumRunFn, label: str, default: bool) -> bool:
    """Shared gum-confirm logic.  Returns ``False`` on non-zero exit (= user said No)."""
    args = ["confirm", "--default=true" if default else "--default=false", label]
    code, _ = run_fn(args)
    return code == 0


def _gum_input_impl(run_fn: GumRunFn, label: str, default: str) -> Optional[str]:
    """Shared gum-input logic.  Returns ``None`` on user cancel (non-zero exit)."""
    args = ["input", "--prompt", f"{label}: ", "--value", default]
    code, output = run_fn(args)
    if code != 0:
        return None
    return output
