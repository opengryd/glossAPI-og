"""
Performance & Power Metrics for the GlossAPI pipeline.

Provides per-phase and end-to-end measurements of:
  * Speed  — Pages Per Second  (PPS)
  * Power  — Pages Per Watt   (PPW) = pages / total_energy_joules

Power is measured from the first available sensor in this priority order:
  1. NVML (pynvml) — per-GPU wattage (CUDA, Linux/Windows)
  2. RAPL           — CPU package energy counters (/sys/class/powercap/, Linux)
  3. powermetrics   — whole-chip power on Apple Silicon (macOS, passwordless sudo only)
  4. None           — timing-only mode; PPW reported as N/A

Usage
-----
::

    from glossapi.perf_metrics import PipelineProfiler, count_pages_for_run

    profiler = PipelineProfiler(output_dir=Path("artifacts/my_run"))

    # Wrap each active processing segment:
    with profiler.measure("extract", backend="docling"):
        extract_pdfs(...)

    pages = count_pages_for_run(output_dir, downloads_dir)
    with profiler.measure("ocr", backend="deepseek-ocr", pages=pages):
        run_ocr(...)

    # Generate & save report:
    report = profiler.report(backend="deepseek-ocr")        # prints table + saves JSON
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page-counting utilities
# ---------------------------------------------------------------------------

def count_pages_from_files(file_paths: "List[Any]") -> int:
    """Count total PDF pages from an iterable of file paths.

    Uses ``pypdfium2`` when available; falls back to ``len(file_paths)``
    (treating each file as 1 page) so timing still works without the lib.
    """
    paths = list(file_paths)
    total = 0
    try:
        import pypdfium2 as _pdfium  # type: ignore

        for p in paths:
            try:
                doc = _pdfium.PdfDocument(str(p))
                try:
                    total += len(doc)
                finally:
                    doc.close()
            except Exception:
                total += 1  # count the document even if page count fails
        return total
    except ImportError:
        pass
    return len(paths)


def count_pages_for_run(output_dir: Path, downloads_dir: Optional[Path] = None) -> int:
    """Return the total number of document pages processed in *output_dir*.

    Strategies (tried in order, highest confidence first):

    1. Sum ``page_count`` values from ``json/metrics/*.metrics.json`` files.
    2. Use ``pypdfium2`` to count pages in every PDF under *downloads_dir*
       (or ``output_dir/downloads/`` when *downloads_dir* is not supplied).
    3. Count ``.md`` files in ``markdown/`` as a last-resort estimate
       (assumes 1 file ≈ an unknown number of pages; logs a warning).

    Returns 0 when no artifact is found.
    """
    total: int = 0

    # Strategy 1 — per-file metrics JSON
    metrics_dir = output_dir / "json" / "metrics"
    if metrics_dir.exists():
        for metrics_path in metrics_dir.glob("*.metrics.json"):
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    pc = data.get("page_count")
                    if pc is not None:
                        total += int(pc)
                        continue
                    pages = data.get("pages")
                    if isinstance(pages, list):
                        total += len(pages)
            except Exception:
                pass
        if total > 0:
            return total

    # Strategy 2 — pypdfium2 on raw PDFs
    dl_dir = downloads_dir or (output_dir / "downloads")
    if dl_dir.exists():
        try:
            import pypdfium2 as _pdfium  # type: ignore

            for pdf in dl_dir.glob("*.pdf"):
                try:
                    doc = _pdfium.PdfDocument(str(pdf))
                    try:
                        total += len(doc)
                    finally:
                        doc.close()
                except Exception:
                    pass
            if total > 0:
                return total
        except ImportError:
            pass

    # Strategy 3 — count markdown files (rough proxy, 1 file = 1 doc, unknown pages)
    md_dir = output_dir / "markdown"
    if md_dir.exists():
        count = len(list(md_dir.glob("*.md")))
        if count > 0:
            logger.warning(
                "perf_metrics: could not determine page count from metrics/PDFs; "
                "using document count (%d) as a rough proxy for pages.",
                count,
            )
            return count

    return 0


# ---------------------------------------------------------------------------
# Power sensors
# ---------------------------------------------------------------------------

class _BasePowerSensor:
    """Abstract base for a power sensor."""

    name: str = "base"

    def available(self) -> bool:
        return False

    def read_joules(self) -> float:
        """Return energy consumed (Joules) since the last call to ``start()``."""
        raise NotImplementedError

    def read_watts(self) -> float:
        """Return instantaneous power draw in Watts."""
        raise NotImplementedError

    def start(self) -> None:
        """Capture the baseline reference before a measurement window."""

    def stop(self) -> Tuple[float, float]:
        """Return (total_energy_joules, average_watts) for the elapsed window."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any resources held by the sensor (e.g. subprocesses).

        Called by :class:`PowerSampler` after sampling stops.  The default
        implementation is a no-op; subclasses with long-lived resources
        (streaming subprocess, open file handles) should override this.
        """


class _NvmlSensor(_BasePowerSensor):
    """NVML-based GPU power sensor using *pynvml* (CUDA on Linux / Windows)."""

    name = "nvml"

    def __init__(self) -> None:
        self._handles: List[Any] = []
        self._nvml: Any = None
        self._initialized = False

    def available(self) -> bool:
        if self._initialized:
            return bool(self._handles)
        self._initialized = True
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._nvml = pynvml
            return bool(self._handles)
        except Exception:
            return False

    def read_watts(self) -> float:
        if not self._handles or self._nvml is None:
            return 0.0
        total = 0.0
        for h in self._handles:
            try:
                # nvmlDeviceGetPowerUsage returns milliwatts
                total += self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
        return total


class _RaplSensor(_BasePowerSensor):
    """RAPL energy counter via Linux ``/sys/class/powercap/``."""

    name = "rapl"
    _PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")

    def available(self) -> bool:
        return self._PATH.exists()

    def _read_uj(self) -> int:
        try:
            return int(self._PATH.read_text().strip())
        except Exception:
            return 0

    def read_watts(self) -> float:
        # Not trivially instantaneous without tracking delta; sampler handles this.
        return 0.0


class _RaplSampledSensor(_BasePowerSensor):
    """RAPL with delta-tracking for the PowerSampler."""

    name = "rapl"
    _PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")

    def __init__(self) -> None:
        self._last_uj: int = 0
        self._last_ts: float = 0.0

    def available(self) -> bool:
        return self._PATH.exists()

    def _read_uj(self) -> int:
        try:
            return int(self._PATH.read_text().strip())
        except Exception:
            return 0

    def start(self) -> None:
        self._last_uj = self._read_uj()
        self._last_ts = time.monotonic()

    def read_watts(self) -> float:
        """Return average watts since the last read."""
        now_uj = self._read_uj()
        now_ts = time.monotonic()
        dt = now_ts - self._last_ts
        if dt <= 0:
            return 0.0
        dU = now_uj - self._last_uj
        if dU < 0:  # counter overflow (rare but possible)
            dU += 2**32
        watts = (dU / 1e6) / dt
        self._last_uj = now_uj
        self._last_ts = now_ts
        return watts


class _IOKitSensor(_BasePowerSensor):
    """No-sudo macOS power reading via ``ioreg`` (Apple Silicon).

    Reads GPU power from the ``AGXPowerMonitor`` IORegistry entry and falls
    back to other known power-reporting services.  Requires no ``sudo`` but
    reflects GPU wattage only, not the full CPU+GPU+ANE package.  Falls back
    gracefully on models where the key is not exposed.

    Values reported by ``ioreg`` are in milliwatts; this class converts to W.
    """

    name = "iokit"

    # (IORegistry service name, candidate property keys) in probe order.
    # Apple Silicon ``ioreg`` properties are reported in milliwatts.
    _PROBES: List[Tuple[str, List[str]]] = [
        ("AGXPowerMonitor", ["GPU Power", "gpu power"]),
        ("IOPlatformEnergyDriver", ["Power", "Total Power", "System Energy"]),
    ]

    def available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        return self._read_mw() is not None

    def _run_ioreg(self, service: str) -> str:
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-n", service],
                capture_output=True, text=True, timeout=2,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def _read_mw(self) -> Optional[float]:
        for service, keys in self._PROBES:
            text = self._run_ioreg(service)
            if not text:
                continue
            for line in text.splitlines():
                line_strip = line.strip()
                for key in keys:
                    if f'"{key}"' in line_strip or f"'{key}'" in line_strip:
                        parts = line_strip.split("=")
                        if len(parts) >= 2:
                            try:
                                val = float(
                                    parts[-1].strip().rstrip(";").split()[0]
                                )
                                if val > 0:
                                    return val
                            except (ValueError, IndexError):
                                pass
        return None

    def read_watts(self) -> float:
        mw = self._read_mw()
        return (mw / 1000.0) if mw is not None else 0.0


class _PowermetricsStreamingSensor(_BasePowerSensor):
    """Apple Silicon full-chip power via a single long-lived ``powermetrics`` subprocess.

    Unlike the previous per-sample approach (which spawned a new process for
    every reading), this class launches ``sudo powermetrics`` **once** on
    :meth:`start` and parses its continuously streaming output in a background
    reader thread.  :meth:`read_watts` simply returns the most recently parsed
    value — no subprocess overhead per sample.

    Covers CPU + GPU + ANE ("Combined Power" line from powermetrics).

    Requires passwordless ``sudo`` for ``powermetrics`` (``sudo -n`` must
    succeed).  The availability check is cached after the first call.
    """

    name = "powermetrics"
    _CMD = ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", "500"]

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._latest_watts: float = 0.0
        self._watts_lock = threading.Lock()
        self._available: Optional[bool] = None
        self._password: Optional[str] = None  # cached from password prompt

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if platform.system() != "Darwin":
            self._available = False
            return False
        # 1) Try passwordless sudo first (already-cached credentials or NOPASSWD).
        try:
            result = subprocess.run(
                ["sudo", "-n", "powermetrics",
                 "--samplers", "cpu_power", "-n", "1", "-i", "1"],
                capture_output=True, text=True, timeout=6,
            )
            if result.returncode == 0:
                self._available = True
                return True
        except Exception:
            pass
        # 2) Prompt the user for a password on the controlling TTY.
        try:
            import getpass as _getpass
            import sys as _sys
            if not _sys.stdin.isatty():
                self._available = False
                return False
            print(
                "\n[GlossAPI] powermetrics needs sudo for energy/power measurement."
                " Leave the password blank to skip power monitoring.",
                flush=True,
            )
            pwd = _getpass.getpass("  sudo password: ")
            if not pwd:
                logger.info("perf_metrics: power monitoring skipped (no password entered).")
                self._available = False
                return False
            # Validate with -S (read password from stdin) and -k (ignore cached ticket).
            test = subprocess.run(
                ["sudo", "-S", "-k", "powermetrics",
                 "--samplers", "cpu_power", "-n", "1", "-i", "1"],
                input=pwd + "\n",
                capture_output=True, text=True, timeout=10,
            )
            if test.returncode == 0:
                self._password = pwd
                self._available = True
            else:
                logger.warning(
                    "perf_metrics: sudo password rejected for powermetrics; "
                    "power monitoring unavailable."
                )
                self._available = False
        except Exception as exc:
            logger.debug("perf_metrics: sudo password prompt failed: %s", exc)
            self._available = False
        return self._available

    def start(self) -> None:
        self._stop_reader.clear()
        self._latest_watts = 0.0
        # Use sudo -S (read password from stdin) when a password was collected at
        # availability-check time; otherwise rely on the cached ticket / NOPASSWD.
        if self._password:
            cmd = ["sudo", "-S", "powermetrics", "--samplers", "cpu_power", "-i", "500"]
        else:
            cmd = self._CMD
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if self._password else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            # Feed the password once so sudo can authenticate, then close stdin.
            if self._password and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(self._password + "\n")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._reader = threading.Thread(
                target=self._parse_loop,
                daemon=True,
                name="PerfPowerMetricsReader",
            )
            self._reader.start()
        except Exception as exc:
            logger.debug(
                "perf_metrics: powermetrics subprocess failed to start: %s", exc
            )
            self._proc = None

    def _parse_loop(self) -> None:
        """Continuously read powermetrics stdout and update ``_latest_watts``."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            for line in self._proc.stdout:
                if self._stop_reader.is_set():
                    break
                line_l = line.lower()
                # Matches: "Combined Power (CPU + GPU + ANE): 5432 mW"
                #      or: "Package Power: 8012 mW"
                if "combined power" in line_l or (
                    "package" in line_l and "power" in line_l
                ):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        tokens = parts[-1].strip().split()
                        if tokens:
                            try:
                                raw = float(tokens[0])
                                unit = tokens[1].lower() if len(tokens) > 1 else "mw"
                                # powermetrics uses mW; guard against W just in case
                                watts = raw / 1000.0 if "mw" in unit else raw
                                with self._watts_lock:
                                    self._latest_watts = watts
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass

    def read_watts(self) -> float:
        with self._watts_lock:
            return self._latest_watts

    def cleanup(self) -> None:
        """Signal the reader to stop and terminate the subprocess."""
        self._stop_reader.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None


_SENSOR_CACHE: Optional[_BasePowerSensor] = None
_SENSOR_CACHE_RESOLVED: bool = False


def _build_sensor() -> Optional[_BasePowerSensor]:
    """Return the best available power sensor or None.

    The result is cached at module level so that interactive prompts (e.g. the
    sudo password prompt for powermetrics) appear only once per process.

    Priority order:

    1. **NVML** — per-GPU wattage via *pynvml* (CUDA, Linux / Windows)
    2. **RAPL** — CPU package energy counter (Linux, no sudo)
    3. **IOKit** — GPU wattage via ``ioreg`` (macOS, no sudo, Apple Silicon)
    4. **powermetrics** — full CPU+GPU+ANE chip power (macOS, sudo with password prompt)
    """
    global _SENSOR_CACHE, _SENSOR_CACHE_RESOLVED
    if _SENSOR_CACHE_RESOLVED:
        return _SENSOR_CACHE
    sensors: List[_BasePowerSensor] = [
        _NvmlSensor(),
        _RaplSampledSensor(),
        _IOKitSensor(),
        _PowermetricsStreamingSensor(),
    ]
    for sensor in sensors:
        try:
            if sensor.available():
                logger.debug("perf_metrics: selected power sensor '%s'", sensor.name)
                _SENSOR_CACHE = sensor
                _SENSOR_CACHE_RESOLVED = True
                return sensor
        except Exception:
            pass
    _SENSOR_CACHE_RESOLVED = True
    return None


# ---------------------------------------------------------------------------
# Background power sampler
# ---------------------------------------------------------------------------

class PowerSampler:
    """Background thread that polls power at *interval_sec* intervals.

    Usage::

        sampler = PowerSampler()
        sampler.start()
        # ... do work ...
        energy_j, source, avg_watts = sampler.stop()
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self._interval = interval_sec
        self._sensor: Optional[_BasePowerSensor] = _build_sensor()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[float] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._stop_time: float = 0.0

    @property
    def source(self) -> str:
        if self._sensor is None:
            return "unavailable"
        return self._sensor.name

    def start(self) -> None:
        """Begin sampling."""
        self._stop_event.clear()
        self._samples.clear()
        self._start_time = time.monotonic()
        if self._sensor is not None:
            try:
                self._sensor.start()
            except Exception:
                pass
        if self._sensor is not None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="PerfPowerSampler")
            self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            if self._sensor is None:
                break
            try:
                w = self._sensor.read_watts()
                with self._lock:
                    self._samples.append(w)
            except Exception:
                pass

    def stop(self) -> Tuple[float, str, Optional[float]]:
        """Stop sampling and return ``(energy_joules, source_name, avg_watts)``.

        Returns ``(0.0, 'unavailable', None)`` when no power sensor is available.
        """
        self._stop_time = time.monotonic()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Release any long-lived resources held by the sensor
        # (e.g. the streaming powermetrics subprocess).
        if self._sensor is not None:
            try:
                self._sensor.cleanup()
            except Exception:
                pass
        if self._sensor is None:
            return 0.0, "unavailable", None

        with self._lock:
            samples = list(self._samples)

        if not samples:
            return 0.0, self._sensor.name, None

        avg_watts = sum(samples) / len(samples)
        elapsed = max(0.0, self._stop_time - self._start_time)
        energy_j = avg_watts * elapsed
        return energy_j, self._sensor.name, avg_watts


# ---------------------------------------------------------------------------
# Phase sample
# ---------------------------------------------------------------------------

@dataclass
class PhaseSample:
    """Measurements captured for a single timed pipeline phase."""

    phase: str
    backend: str
    active_sec: float
    pages: int
    energy_joules: Optional[float]
    avg_watts: Optional[float]
    power_source: str

    # --- derived metrics ---

    @property
    def pps(self) -> Optional[float]:
        """Pages per second (PPS)."""
        if self.active_sec <= 0 or self.pages <= 0:
            return None
        return self.pages / self.active_sec

    @property
    def ppw(self) -> Optional[float]:
        """Pages per Watt-second (PPW = pages / energy_joules)."""
        if self.energy_joules is None or self.energy_joules <= 0 or self.pages <= 0:
            return None
        return self.pages / self.energy_joules

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["pps"] = self.pps
        base["ppw"] = self.ppw
        return base


# ---------------------------------------------------------------------------
# Pipeline profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Accumulates phase measurements for one or more pipeline phases.

    Typically one profiler lives on a ``Corpus`` instance and records every
    ``extract()``, ``clean()``, and ``ocr()`` call.  After the pipeline
    completes, call :meth:`report` to get a structured dict and an auto-saved
    JSON file under ``output_dir/logs/``.

    Thread-safety: individual :meth:`measure` contexts serialize via an
    internal lock; parallel use across threads is supported.
    """

    def __init__(self, output_dir: Path, poll_interval_sec: float = 0.5) -> None:
        self.output_dir = Path(output_dir)
        self._poll_interval = poll_interval_sec
        self._samples: List[PhaseSample] = []
        self._lock = threading.Lock()
        self._run_id: str = uuid.uuid4().hex[:12]

    def reset(self) -> None:
        """Clear all accumulated samples (call before a fresh pipeline run)."""
        with self._lock:
            self._samples.clear()
            self._run_id = uuid.uuid4().hex[:12]

    def record_sample(
        self,
        phase: str,
        *,
        active_sec: float,
        pages: int,
        energy_joules: Optional[float],
        avg_watts: Optional[float],
        power_source: str,
        backend: str = "unknown",
    ) -> None:
        """Record a pre-computed phase sample directly.

        Use this when you need manual start/stop control (e.g. wrapping code
        that spans a try/finally).  Prefer :meth:`measure` when possible.
        """
        sample = PhaseSample(
            phase=phase,
            backend=backend,
            active_sec=active_sec,
            pages=pages,
            energy_joules=energy_joules,
            avg_watts=avg_watts,
            power_source=power_source,
        )
        with self._lock:
            self._samples.append(sample)

        logger.info(
            "perf_metrics [%s/%s]: active=%.2fs  pages=%d  "
            "pps=%s  energy=%.1fJ  ppw=%s  power_source=%s",
            backend,
            phase,
            active_sec,
            pages,
            f"{sample.pps:.3f}" if sample.pps is not None else "N/A",
            energy_joules if energy_joules is not None else 0.0,
            f"{sample.ppw:.4f}" if sample.ppw is not None else "N/A",
            power_source,
        )

    @contextmanager
    def measure(
        self,
        phase: str,
        *,
        pages: int = 0,
        backend: str = "unknown",
    ) -> Generator[None, None, None]:
        """Context manager that times *phase* and samples power.

        Parameters
        ----------
        phase:
            Human-readable phase label, e.g. ``"extract"``, ``"ocr"``, ``"clean"``.
        pages:
            Number of document pages processed inside this block.  When 0,
            the profiler still records timing but PPS/PPW will be N/A.
        backend:
            Backend label, e.g. ``"deepseek-ocr"``, ``"rapidocr"``.
        """
        sampler = PowerSampler(interval_sec=self._poll_interval)
        sampler.start()
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            energy_j, source, avg_watts = sampler.stop()

            sample = PhaseSample(
                phase=phase,
                backend=backend,
                active_sec=elapsed,
                pages=pages,
                energy_joules=energy_j if source != "unavailable" else None,
                avg_watts=avg_watts,
                power_source=source,
            )
            with self._lock:
                self._samples.append(sample)

            logger.info(
                "perf_metrics [%s/%s]: active=%.2fs  pages=%d  "
                "pps=%s  energy=%.1fJ  ppw=%s  power_source=%s",
                backend,
                phase,
                elapsed,
                pages,
                f"{sample.pps:.3f}" if sample.pps is not None else "N/A",
                energy_j if source != "unavailable" else 0.0,
                f"{sample.ppw:.4f}" if sample.ppw is not None else "N/A",
                source,
            )

    def report(self, *, backend: Optional[str] = None) -> Dict[str, Any]:
        """Build and persist a Performance & Power Report.

        Parameters
        ----------
        backend:
            Override backend label used in the report filename.  Defaults to
            the backend of the last recorded sample (or ``"unknown"``).

        Returns the report dict.  Also writes a JSON file to
        ``output_dir/logs/perf_report_<backend>_<timestamp>.json`` and prints
        a summary table to the logger.
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            logger.warning("perf_metrics.report(): no samples recorded — nothing to report.")
            return {}

        # Resolve backend label
        report_backend = backend or samples[-1].backend

        # ---- Aggregate phases ----
        phase_reports: Dict[str, Dict[str, Any]] = {}
        total_active_sec = 0.0
        total_pages = 0
        total_energy_j = 0.0
        has_power = False
        power_source = "unavailable"
        avg_watts_list: List[float] = []

        for s in samples:
            total_active_sec += s.active_sec
            if total_pages < s.pages:
                total_pages = s.pages  # take largest (phases share the same doc set)
            if s.energy_joules is not None:
                total_energy_j += s.energy_joules
                has_power = True
                power_source = s.power_source
            if s.avg_watts is not None:
                avg_watts_list.append(s.avg_watts)

            phase_reports[s.phase] = {
                "active_sec": round(s.active_sec, 3),
                "pages": s.pages,
                "pps": round(s.pps, 4) if s.pps is not None else None,
                "energy_joules": round(s.energy_joules, 3) if s.energy_joules is not None else None,
                "avg_watts": round(s.avg_watts, 2) if s.avg_watts is not None else None,
                "ppw": round(s.ppw, 6) if s.ppw is not None else None,
            }

        # End-to-end aggregates
        e2e_pps = (total_pages / total_active_sec) if total_active_sec > 0 and total_pages > 0 else None
        e2e_energy = total_energy_j if has_power else None
        e2e_ppw = (total_pages / total_energy_j) if total_energy_j > 0 and total_pages > 0 else None
        e2e_avg_watts = (sum(avg_watts_list) / len(avg_watts_list)) if avg_watts_list else None

        report: Dict[str, Any] = {
            "run_id": self._run_id,
            "backend": report_backend,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "total_pages": total_pages,
            "power_source": power_source,
            "phases": phase_reports,
            "end_to_end": {
                "total_active_sec": round(total_active_sec, 3),
                "pps": round(e2e_pps, 4) if e2e_pps is not None else None,
                "energy_joules": round(e2e_energy, 3) if e2e_energy is not None else None,
                "avg_watts": round(e2e_avg_watts, 2) if e2e_avg_watts is not None else None,
                "ppw": round(e2e_ppw, 6) if e2e_ppw is not None else None,
            },
        }

        # ---- Persist JSON ----
        self._save_report(report, backend_label=report_backend)

        # ---- Console summary ----
        self._log_report(report)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_report(self, report: Dict[str, Any], backend_label: str) -> None:
        """Save *report* to ``output_dir/logs/perf_report_<backend>_<ts>.json``."""
        logs_dir = self.output_dir / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            safe_backend = backend_label.replace("/", "-").replace(" ", "_")
            ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            filename = f"perf_report_{safe_backend}_{ts}.json"
            out_path = logs_dir / filename
            tmp_path = out_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp_path, out_path)
            logger.info("perf_metrics: report saved → %s", out_path)
        except Exception as exc:
            logger.warning("perf_metrics: failed to save report: %s", exc)

    def _log_report(self, report: Dict[str, Any]) -> None:
        """Log a human-readable summary table."""
        backend = report.get("backend", "?")
        total_pages = report.get("total_pages", 0)
        power_source = report.get("power_source", "unavailable")
        e2e = report.get("end_to_end", {})

        sep = "-" * 78
        lines: List[str] = [
            "",
            sep,
            f"  PERFORMANCE & POWER REPORT — backend: {backend}",
            sep,
            f"  Total pages    : {total_pages}",
            f"  Power source   : {power_source}",
            "",
            f"  {'Phase':<20} {'Active (s)':>12} {'PPS':>10} {'Energy (J)':>12} {'Avg W':>9} {'PPW':>14}",
            f"  {'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*9} {'-'*14}",
        ]

        for phase_name, ph in report.get("phases", {}).items():
            active = ph.get("active_sec", 0.0)
            pps = ph.get("pps")
            energy = ph.get("energy_joules")
            avg_w = ph.get("avg_watts")
            ppw = ph.get("ppw")
            lines.append(
                f"  {phase_name:<20} {active:>12.2f} "
                f"{_fmt(pps, '.3f'):>10} "
                f"{_fmt(energy, '.1f'):>12} "
                f"{_fmt(avg_w, '.1f'):>9} "
                f"{_fmt(ppw, '.5f'):>14}"
            )

        lines += [
            f"  {'':20} {'-'*12} {'-'*10} {'-'*12} {'-'*9} {'-'*14}",
            f"  {'END-TO-END':<20} "
            f"{e2e.get('total_active_sec', 0.0):>12.2f} "
            f"{_fmt(e2e.get('pps'), '.3f'):>10} "
            f"{_fmt(e2e.get('energy_joules'), '.1f'):>12} "
            f"{_fmt(e2e.get('avg_watts'), '.1f'):>9} "
            f"{_fmt(e2e.get('ppw'), '.5f'):>14}",
            sep,
            "",
        ]
        for ln in lines:
            logger.info(ln)


def _fmt(val: Any, fmt: str) -> str:
    """Format *val* using *fmt* or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)
