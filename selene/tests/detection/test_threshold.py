"""Tests for ThresholdDetector."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from selene.core.interfaces import (
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
    TelemetryWindow,
)
from selene.detection.threshold import ThresholdDetector

CONFIG_YAML = Path(__file__).parent.parent.parent / "config" / "sensor_ranges.yaml"

_T0 = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_window(sensor_id: str, values: list[float], unit: str = "bar") -> TelemetryWindow:
    frames = []
    for i, v in enumerate(values):
        from datetime import timedelta
        ts = _T0 + i * timedelta(minutes=5)
        reading = SensorReading(sensor_id=sensor_id, timestamp=ts, value=v, unit=unit)
        frames.append(TelemetryFrame(timestamp=ts, readings={sensor_id: reading}))
    return TelemetryWindow(frames=frames, start=frames[0].timestamp, end=frames[-1].timestamp)


def _make_meta(sensor_id: str, sensor_type: str, unit: str = "") -> SensorMetadata:
    return SensorMetadata(
        sensors={sensor_id: {"unit": unit, "subsystem": "TCS", "sensor_type": sensor_type}},
        subsystems=["TCS"],
        sampling_rate_seconds=300.0,
    )


class TestThresholdDetectorDirect:
    def test_no_events_when_all_in_range(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        window = _make_window("s", [1.0, 5.0, 9.9], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert events == []

    def test_event_when_exceeds_max(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        window = _make_window("s", [5.0, 12.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert len(events) == 1
        assert events[0].affected_sensors == ["s"]
        assert events[0].detector_name == "threshold"

    def test_event_when_below_min(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        window = _make_window("s", [-3.0, 5.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert len(events) == 1
        assert events[0].score == pytest.approx(0.3)

    def test_score_normalized_by_range_width(self):
        # range [0, 100], value = 110 → score = 10/100 = 0.1
        det = ThresholdDetector({"s": (0.0, 100.0)})
        window = _make_window("s", [110.0], unit="ppm")
        events = asyncio.run(det.evaluate(window))
        assert len(events) == 1
        assert events[0].score == pytest.approx(0.1)

    def test_multiple_violations_in_window(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        window = _make_window("s", [11.0, 5.0, 12.0, 13.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert len(events) == 3

    def test_details_populated(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        window = _make_window("s", [15.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        d = events[0].details
        assert d["value"] == 15.0
        assert d["range_min"] == 0.0
        assert d["range_max"] == 10.0

    def test_sensor_absent_from_ranges_skipped(self):
        det = ThresholdDetector({"other": (0.0, 10.0)})
        window = _make_window("s", [999.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert events == []

    def test_sensor_not_in_frame_skipped(self):
        det = ThresholdDetector({"s": (0.0, 10.0)})
        # frame has "other", not "s"
        reading = SensorReading(sensor_id="other", timestamp=_T0, value=999.0, unit="bar")
        frame = TelemetryFrame(timestamp=_T0, readings={"other": reading})
        window = TelemetryWindow(frames=[frame], start=_T0, end=_T0)
        events = asyncio.run(det.evaluate(window))
        assert events == []


class TestThresholdDetectorFromYaml:
    def test_loads_type_defaults(self):
        meta = _make_meta("tcs/pressure-ams", "P", "bar")
        det = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=meta)
        # tcs/pressure-ams has a sensor_override in the YAML
        assert "tcs/pressure-ams" in det._ranges
        lo, hi = det._ranges["tcs/pressure-ams"]
        assert lo < hi

    def test_sensor_override_takes_precedence(self):
        meta = _make_meta("tcs/pressure-ams", "P", "bar")
        det_override = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=meta)
        # The override for tcs/pressure-ams is [0.2, 3.5], tighter than P default [-8, 5]
        lo, hi = det_override._ranges["tcs/pressure-ams"]
        assert lo == pytest.approx(0.2)
        assert hi == pytest.approx(3.5)

    def test_no_metadata_uses_only_overrides(self):
        det = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=None)
        # Only sensor_overrides entries should be present
        assert "tcs/pressure-ams" in det._ranges
        # A non-override sensor should be absent
        assert "ams-feg/co2-1" not in det._ranges

    def test_value_outside_yaml_range_triggers_event(self):
        meta = _make_meta("tcs/pressure-ams", "P", "bar")
        det = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=meta)
        # Override max is 3.5; inject value of 5.0
        window = _make_window("tcs/pressure-ams", [5.0], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert len(events) == 1

    def test_value_inside_yaml_range_no_event(self):
        meta = _make_meta("tcs/pressure-ams", "P", "bar")
        det = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=meta)
        window = _make_window("tcs/pressure-ams", [1.5], unit="bar")
        events = asyncio.run(det.evaluate(window))
        assert events == []

    def test_co2_sensor_uses_type_default(self):
        meta = _make_meta("ams-feg/co2-1", "CO2", "ppm")
        det = ThresholdDetector.from_yaml(CONFIG_YAML, metadata=meta)
        assert "ams-feg/co2-1" in det._ranges
        lo, hi = det._ranges["ams-feg/co2-1"]
        assert lo == pytest.approx(400.0)
        assert hi == pytest.approx(2500.0)
