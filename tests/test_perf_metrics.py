"""
Tests for src/glossapi/perf_metrics.py

Covers:
  - count_pages_from_files (with and without pypdfium2)
  - count_pages_for_run (metrics JSON, pypdfium2, markdown fallback)
  - PowerSampler with no sensor vs. mock sensor
  - PhaseSample PPS / PPW properties
  - PipelineProfiler.measure() context manager
  - PipelineProfiler.record_sample() manual path
  - PipelineProfiler.report() structure
  - PipelineProfiler.reset() clears state
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

from glossapi.perf_metrics import (
    PhaseSample,
    PipelineProfiler,
    PowerSampler,
    _IOKitSensor,
    _PowermetricsStreamingSensor,
    _build_sensor,
    _fmt,
    count_pages_from_files,
    count_pages_for_run,
)


# ===========================================================================
# count_pages_from_files
# ===========================================================================

class TestCountPagesFromFiles:
    def test_empty_list_returns_zero(self):
        assert count_pages_from_files([]) == 0

    def test_fallback_when_pypdfium2_missing(self, tmp_path, monkeypatch):
        """When pypdfium2 is not importable, falls back to len(paths)."""
        fake_files = [tmp_path / f"doc{i}.pdf" for i in range(5)]
        for f in fake_files:
            f.write_bytes(b"")  # empty, non-PDF bytes — just needs to exist

        # Make pypdfium2 unavailable
        monkeypatch.setitem(sys.modules, "pypdfium2", None)
        result = count_pages_from_files(fake_files)
        assert result == 5

    def test_pypdfium2_success(self, tmp_path):
        """When pypdfium2 is present, use page counts from PdfDocument."""
        fake_doc = MagicMock()
        # Simulate a 3-page document
        fake_doc.__len__ = MagicMock(return_value=3)
        fake_doc.close = MagicMock()

        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.return_value = fake_doc

        with patch.dict(sys.modules, {"pypdfium2": fake_pdfium}):
            result = count_pages_from_files([Path("a.pdf"), Path("b.pdf")])

        assert result == 6  # 2 docs × 3 pages each

    def test_pypdfium2_per_file_exception_counts_doc_as_one(self, tmp_path):
        """A per-file exception should still count that file as 1 page."""
        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.side_effect = RuntimeError("bad pdf")

        with patch.dict(sys.modules, {"pypdfium2": fake_pdfium}):
            result = count_pages_from_files([Path("bad.pdf"), Path("bad2.pdf")])

        assert result == 2  # both counted as 1


# ===========================================================================
# count_pages_for_run
# ===========================================================================

class TestCountPagesForRun:
    def _make_metrics(self, metrics_dir: Path, stem: str, pages: int) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{stem}.metrics.json").write_text(
            json.dumps({"page_count": pages}), encoding="utf-8"
        )

    def test_returns_zero_when_no_artifacts(self, tmp_path):
        result = count_pages_for_run(tmp_path)
        assert result == 0

    def test_strategy1_metrics_json(self, tmp_path):
        """Should sum page_count from *.metrics.json files."""
        metrics_dir = tmp_path / "json" / "metrics"
        self._make_metrics(metrics_dir, "doc1", 12)
        self._make_metrics(metrics_dir, "doc2", 7)
        assert count_pages_for_run(tmp_path) == 19

    def test_strategy1_pages_list_fallback(self, tmp_path):
        """When page_count is absent, fall back to len(pages) list."""
        metrics_dir = tmp_path / "json" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "doc1.metrics.json").write_text(
            json.dumps({"pages": [1, 2, 3, 4, 5]}), encoding="utf-8"
        )
        assert count_pages_for_run(tmp_path) == 5

    def test_strategy3_markdown_file_count(self, tmp_path):
        """Last resort: count .md files when metrics and PDFs are absent."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        for i in range(4):
            (md_dir / f"doc{i}.md").write_text("# content", encoding="utf-8")
        # strategy 1 and 2 must fail
        result = count_pages_for_run(tmp_path)
        assert result == 4

    def test_strategy1_takes_precedence_over_strategy3(self, tmp_path):
        """Metrics JSON result wins over markdown count."""
        metrics_dir = tmp_path / "json" / "metrics"
        self._make_metrics(metrics_dir, "doc1", 100)
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "doc1.md").write_text("x")
        assert count_pages_for_run(tmp_path) == 100

    def test_strategy2_pypdfium2(self, tmp_path):
        """Strategy 2 uses pypdfium2 on PDFs in downloads/."""
        dl_dir = tmp_path / "downloads"
        dl_dir.mkdir()
        (dl_dir / "a.pdf").write_bytes(b"")
        (dl_dir / "b.pdf").write_bytes(b"")

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=5)
        fake_doc.close = MagicMock()
        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.return_value = fake_doc

        with patch.dict(sys.modules, {"pypdfium2": fake_pdfium}):
            result = count_pages_for_run(tmp_path)
        assert result == 10  # 2 PDFs × 5 pages

    def test_corrupt_metrics_json_ignored(self, tmp_path):
        """Malformed JSON files are silently skipped."""
        metrics_dir = tmp_path / "json" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "bad.metrics.json").write_text("{not json}", encoding="utf-8")
        # Falls through to strategy 3 (no markdown → 0)
        assert count_pages_for_run(tmp_path) == 0


# ===========================================================================
# PhaseSample
# ===========================================================================

class TestPhaseSample:
    def _make(self, pages=10, active_sec=5.0, energy_joules=50.0, avg_watts=10.0) -> PhaseSample:
        return PhaseSample(
            phase="extract",
            backend="docling",
            active_sec=active_sec,
            pages=pages,
            energy_joules=energy_joules,
            avg_watts=avg_watts,
            power_source="nvml",
        )

    def test_pps_calculation(self):
        s = self._make(pages=10, active_sec=5.0)
        assert abs(s.pps - 2.0) < 1e-9

    def test_ppw_calculation(self):
        s = self._make(pages=10, energy_joules=50.0)
        assert abs(s.ppw - 0.2) < 1e-9

    def test_pps_none_when_zero_pages(self):
        assert self._make(pages=0).pps is None

    def test_pps_none_when_zero_time(self):
        assert self._make(active_sec=0.0).pps is None

    def test_ppw_none_when_zero_energy(self):
        assert self._make(energy_joules=0.0).ppw is None

    def test_ppw_none_when_energy_is_none(self):
        s = self._make()
        s.energy_joules = None
        assert s.ppw is None

    def test_ppw_none_when_zero_pages(self):
        assert self._make(pages=0, energy_joules=50.0).ppw is None

    def test_to_dict_includes_pps_ppw(self):
        s = self._make(pages=10, active_sec=2.0, energy_joules=20.0)
        d = s.to_dict()
        assert "pps" in d
        assert "ppw" in d
        assert abs(d["pps"] - 5.0) < 1e-9


# ===========================================================================
# PowerSampler (no real sensor)
# ===========================================================================

class TestPowerSamplerNoSensor:
    def test_returns_unavailable_without_sensor(self):
        """With _sensor=None, stop() returns (0.0, 'unavailable', None)."""
        sampler = PowerSampler()
        sampler._sensor = None  # force no sensor
        sampler.start()
        time.sleep(0.05)
        energy, source, avg = sampler.stop()
        assert energy == 0.0
        assert source == "unavailable"
        assert avg is None

    def test_source_property_without_sensor(self):
        sampler = PowerSampler()
        sampler._sensor = None
        assert sampler.source == "unavailable"


class TestPowerSamplerWithMockSensor:
    """Tests with a mock sensor that returns a fixed wattage."""

    def _make_sampler(self, watts: float = 100.0) -> PowerSampler:
        sensor = MagicMock()
        sensor.name = "mock"
        sensor.available.return_value = True
        sensor.start.return_value = None
        sensor.read_watts.return_value = watts
        sampler = PowerSampler(interval_sec=0.05)
        sampler._sensor = sensor
        return sampler

    def test_returns_positive_energy(self):
        sampler = self._make_sampler(watts=100.0)
        sampler.start()
        time.sleep(0.2)
        energy, source, avg = sampler.stop()
        assert source == "mock"
        assert avg is not None
        assert abs(avg - 100.0) < 1e-9
        assert energy > 0.0

    def test_energy_roughly_watts_times_time(self):
        """energy_J ≈ avg_watts × elapsed_seconds — allow 30% tolerance."""
        watts = 50.0
        sleep_sec = 0.3
        sampler = self._make_sampler(watts=watts)
        sampler.start()
        time.sleep(sleep_sec)
        energy, _, avg = sampler.stop()
        expected = watts * sleep_sec
        assert energy == pytest.approx(expected, rel=0.40)  # generous tolerance for CI

    def test_source_property_with_sensor(self):
        sampler = self._make_sampler()
        assert sampler.source == "mock"


# ===========================================================================
# _IOKitSensor
# ===========================================================================

class TestIOKitSensor:
    def test_unavailable_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        sensor = _IOKitSensor()
        assert sensor.available() is False

    def test_read_watts_parses_gpu_power_line(self, monkeypatch):
        """read_watts() converts milliwatts from ioreg to watts."""
        ioreg_output = (
            '| o AGXPowerMonitor  <class ...>\n'
            '|   {\n'
            '|     "GPU Power" = 2450\n'
            '|   }\n'
        )
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        sensor = _IOKitSensor()
        monkeypatch.setattr(sensor, "_run_ioreg", lambda svc: ioreg_output if svc == "AGXPowerMonitor" else "")
        assert abs(sensor.read_watts() - 2.45) < 1e-9

    def test_read_watts_returns_zero_when_no_match(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        sensor = _IOKitSensor()
        monkeypatch.setattr(sensor, "_run_ioreg", lambda svc: "no power data here\n")
        assert sensor.read_watts() == 0.0

    def test_available_false_when_read_returns_none(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        sensor = _IOKitSensor()
        monkeypatch.setattr(sensor, "_read_mw", lambda: None)
        assert sensor.available() is False

    def test_available_true_when_read_returns_value(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        sensor = _IOKitSensor()
        monkeypatch.setattr(sensor, "_read_mw", lambda: 1500.0)
        assert sensor.available() is True

    def test_cleanup_is_noop(self):
        """_IOKitSensor.cleanup() should not raise."""
        sensor = _IOKitSensor()
        sensor.cleanup()  # should not raise


# ===========================================================================
# _PowermetricsStreamingSensor
# ===========================================================================

class TestPowermetricsStreamingSensor:
    def test_unavailable_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        sensor = _PowermetricsStreamingSensor()
        assert sensor.available() is False

    def test_available_caches_result(self, monkeypatch):
        """available() result is cached after first call."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        sensor = _PowermetricsStreamingSensor()
        sensor._available = True
        # Even if the system call would fail, it returns the cached value
        assert sensor.available() is True

    def test_parse_loop_extracts_combined_power(self, tmp_path):
        """_parse_loop() correctly parses 'Combined Power' lines in mW."""
        import io
        sensor = _PowermetricsStreamingSensor()
        # Simulate powermetrics stdout with a combined power line
        fake_stdout = io.StringIO(
            "*** Sampled system activity (Thu Feb 24 ...)\n"
            "Combined Power (CPU + GPU + ANE): 4256 mW\n"
        )
        # Directly feed stdout to the parser via a mock Popen
        mock_proc = MagicMock()
        mock_proc.stdout = fake_stdout
        sensor._proc = mock_proc
        sensor._parse_loop()
        assert abs(sensor._latest_watts - 4.256) < 1e-6

    def test_parse_loop_extracts_package_power(self):
        """_parse_loop() falls back to 'Package Power' lines."""
        import io
        sensor = _PowermetricsStreamingSensor()
        fake_stdout = io.StringIO("Package Power: 6100 mW\n")
        sensor._proc = MagicMock()
        sensor._proc.stdout = fake_stdout
        sensor._parse_loop()
        assert abs(sensor._latest_watts - 6.1) < 1e-6

    def test_read_watts_returns_latest(self):
        sensor = _PowermetricsStreamingSensor()
        sensor._latest_watts = 12.5
        assert sensor.read_watts() == 12.5

    def test_cleanup_handles_no_subprocess(self):
        """cleanup() with no subprocess should not raise."""
        sensor = _PowermetricsStreamingSensor()
        sensor.cleanup()  # _proc is None

    def test_cleanup_terminates_subprocess(self):
        """cleanup() calls terminate() on the subprocess."""
        sensor = _PowermetricsStreamingSensor()
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        sensor._proc = mock_proc
        sensor.cleanup()
        mock_proc.terminate.assert_called_once()
        assert sensor._proc is None


# ===========================================================================
# PowerSampler cleanup
# ===========================================================================

class TestPowerSamplerCleanup:
    def test_stop_calls_sensor_cleanup(self):
        """PowerSampler.stop() must call sensor.cleanup() to release resources."""
        sampler = PowerSampler()
        mock_sensor = MagicMock()
        mock_sensor.name = "mock"
        mock_sensor.read_watts.return_value = 10.0
        mock_sensor.start.return_value = None
        sampler._sensor = mock_sensor
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        mock_sensor.cleanup.assert_called_once()

    def test_stop_does_not_raise_if_cleanup_errors(self):
        """cleanup() errors must be swallowed, not propagated."""
        sampler = PowerSampler()
        mock_sensor = MagicMock()
        mock_sensor.name = "mock"
        mock_sensor.cleanup.side_effect = RuntimeError("boom")
        mock_sensor.read_watts.return_value = 5.0
        sampler._sensor = mock_sensor
        sampler.start()
        time.sleep(0.05)
        # Should NOT raise despite cleanup() throwing
        energy, source, avg = sampler.stop()
        assert source == "mock"


# ===========================================================================
# PipelineProfiler
# ===========================================================================

class TestPipelineProfilerMeasure:
    def test_measure_records_sample(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        with prof.measure("extract", pages=20, backend="docling"):
            time.sleep(0.05)
        assert len(prof._samples) == 1
        s = prof._samples[0]
        assert s.phase == "extract"
        assert s.backend == "docling"
        assert s.pages == 20
        assert s.active_sec >= 0.04

    def test_measure_records_multiple_phases(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        with prof.measure("extract", pages=5, backend="docling"):
            time.sleep(0.02)
        with prof.measure("ocr", pages=5, backend="deepseek-ocr"):
            time.sleep(0.02)
        assert len(prof._samples) == 2
        phases = {s.phase for s in prof._samples}
        assert phases == {"extract", "ocr"}

    def test_measure_still_records_on_exception(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        with pytest.raises(ValueError):
            with prof.measure("clean", pages=3, backend="rust"):
                raise ValueError("deliberate")
        # Sample should still have been recorded via finally
        assert len(prof._samples) == 1

    def test_measure_pps_positive(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        with prof.measure("extract", pages=10, backend="docling"):
            time.sleep(0.05)
        s = prof._samples[0]
        assert s.pps is not None
        assert s.pps > 0


class TestPipelineProfilerRecordSample:
    def test_record_sample_manual(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        prof.record_sample(
            "clean",
            active_sec=3.0,
            pages=30,
            energy_joules=15.0,
            avg_watts=5.0,
            power_source="nvml",
            backend="rust-cleaner",
        )
        assert len(prof._samples) == 1
        s = prof._samples[0]
        assert s.phase == "clean"
        assert s.backend == "rust-cleaner"
        assert abs(s.pps - 10.0) < 1e-9
        assert abs(s.ppw - 2.0) < 1e-9


class TestPipelineProfilerReport:
    def _prof_with_samples(self, tmp_path) -> PipelineProfiler:
        prof = PipelineProfiler(output_dir=tmp_path)
        prof.record_sample(
            "extract", active_sec=10.0, pages=100, energy_joules=100.0,
            avg_watts=10.0, power_source="nvml", backend="docling",
        )
        prof.record_sample(
            "ocr", active_sec=50.0, pages=100, energy_joules=500.0,
            avg_watts=10.0, power_source="nvml", backend="deepseek-ocr",
        )
        return prof

    def test_report_has_required_top_level_keys(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        report = prof.report(backend="deepseek-ocr")
        for key in ("run_id", "backend", "timestamp", "total_pages", "power_source",
                    "phases", "end_to_end"):
            assert key in report, f"missing key: {key}"

    def test_report_phases_present(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        report = prof.report()
        assert "extract" in report["phases"]
        assert "ocr" in report["phases"]

    def test_report_end_to_end_pps(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        report = prof.report()
        e2e = report["end_to_end"]
        # total_active_sec = 10 + 50 = 60; total_pages = 100 (max of phases)
        assert abs(e2e["total_active_sec"] - 60.0) < 1e-3
        # PPS = 100/60 ≈ 1.6667
        assert e2e["pps"] == pytest.approx(100 / 60, rel=1e-3)

    def test_report_end_to_end_ppw(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        report = prof.report()
        e2e = report["end_to_end"]
        # total_energy = 600 J; PPW = 100/600
        assert e2e["ppw"] == pytest.approx(100 / 600, rel=1e-3)

    def test_report_backend_override(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        report = prof.report(backend="custom-backend")
        assert report["backend"] == "custom-backend"

    def test_report_empty_returns_empty_dict(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        report = prof.report()
        assert report == {}

    def test_report_saves_json_file(self, tmp_path):
        prof = self._prof_with_samples(tmp_path)
        prof.report(backend="test-backend")
        logs_dir = tmp_path / "logs"
        json_files = list(logs_dir.glob("perf_report_test-backend_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["backend"] == "test-backend"

    def test_report_phase_pps_ppw_values(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        prof.record_sample(
            "extract", active_sec=10.0, pages=50,
            energy_joules=100.0, avg_watts=10.0,
            power_source="nvml", backend="docling",
        )
        report = prof.report()
        ph = report["phases"]["extract"]
        assert ph["pps"] == pytest.approx(5.0, rel=1e-4)
        assert ph["ppw"] == pytest.approx(0.5, rel=1e-4)

    def test_report_no_power_source_yields_none_ppw(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        prof.record_sample(
            "extract", active_sec=5.0, pages=10,
            energy_joules=None, avg_watts=None,
            power_source="unavailable", backend="docling",
        )
        report = prof.report()
        e2e = report["end_to_end"]
        assert e2e["ppw"] is None
        assert e2e["energy_joules"] is None


class TestPipelineProfilerReset:
    def test_reset_clears_samples(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        with prof.measure("extract", pages=5, backend="docling"):
            pass
        assert len(prof._samples) == 1
        prof.reset()
        assert len(prof._samples) == 0

    def test_reset_generates_new_run_id(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        original_id = prof._run_id
        prof.reset()
        assert prof._run_id != original_id

    def test_reset_report_returns_empty_after_reset(self, tmp_path):
        prof = PipelineProfiler(output_dir=tmp_path)
        prof.record_sample(
            "extract", active_sec=1.0, pages=5,
            energy_joules=10.0, avg_watts=10.0,
            power_source="nvml", backend="test",
        )
        prof.reset()
        assert prof.report() == {}


# ===========================================================================
# Thread safety
# ===========================================================================

class TestPipelineProfilerThreadSafety:
    def test_concurrent_measure_all_recorded(self, tmp_path):
        """Multiple threads writing samples concurrently should all be recorded."""
        prof = PipelineProfiler(output_dir=tmp_path, poll_interval_sec=0.5)

        errors = []

        def worker(i):
            try:
                prof.record_sample(
                    f"phase_{i}", active_sec=0.1 * i, pages=i * 2,
                    energy_joules=float(i), avg_watts=float(i),
                    power_source="mock", backend="worker",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 21)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(prof._samples) == 20


# ===========================================================================
# _fmt helper
# ===========================================================================

class TestFmtHelper:
    def test_none_returns_na(self):
        assert _fmt(None, ".2f") == "N/A"

    def test_float_formatted(self):
        assert _fmt(3.14159, ".2f") == "3.14"

    def test_int_formatted(self):
        assert _fmt(42, "d") == "42"

    def test_invalid_format_returns_str(self):
        assert _fmt("hello", ".2f") == "hello"


# ===========================================================================
# Public API exports from glossapi
# ===========================================================================

class TestPublicExports:
    def test_pipeline_profiler_importable(self):
        from glossapi import PipelineProfiler as PP
        assert PP is PipelineProfiler

    def test_count_pages_for_run_importable(self):
        from glossapi import count_pages_for_run as cpfr
        assert cpfr is count_pages_for_run

    def test_count_pages_from_files_importable(self):
        from glossapi import count_pages_from_files as cpff
        assert cpff is count_pages_from_files
