"""Tests for DampDetector."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from selene.core.interfaces import AnomalyEvent, SensorReading, TelemetryFrame, TelemetryWindow
from selene.detection.damp import DampDetector

_T0 = datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
_DT = timedelta(minutes=5)


def _make_window(series: dict[str, list[float]]) -> TelemetryWindow:
    """Build a TelemetryWindow from {sensor_id: [values]} — one frame per sample."""
    sensor_ids = list(series)
    n = len(next(iter(series.values())))
    frames: list[TelemetryFrame] = []
    for i in range(n):
        ts = _T0 + i * _DT
        readings = {
            sid: SensorReading(sensor_id=sid, timestamp=ts, value=series[sid][i], unit="")
            for sid in sensor_ids
        }
        frames.append(TelemetryFrame(timestamp=ts, readings=readings))
    return TelemetryWindow(frames=frames, start=frames[0].timestamp, end=frames[-1].timestamp)


def _sine_with_discord(n: int = 200, discord_idx: int = 150, discord_mag: float = 10.0) -> list[float]:
    """Sine wave with a single-point step discord injected."""
    t = np.linspace(0, 4 * math.pi, n)
    signal = np.sin(t).tolist()
    # Replace a short window at discord_idx with a flat offset pulse
    for j in range(discord_idx, min(discord_idx + 8, n)):
        signal[j] += discord_mag
    return signal


class TestDampDetectorDetection:
    def test_flags_known_discord(self):
        signal = _sine_with_discord(n=200, discord_idx=150, discord_mag=10.0)
        window = _make_window({"sensor_a": signal})
        detector = DampDetector(sensor_ids=["sensor_a"], window_length=20, threshold=1.0)
        events = pytest.run_async(detector.evaluate(window)) if False else None

        import asyncio
        events = asyncio.run(detector.evaluate(window))

        assert len(events) == 1
        assert events[0].detector_name == "damp"
        assert events[0].affected_sensors == ["sensor_a"]
        assert events[0].score > 1.0

    def test_no_event_on_clean_signal(self):
        import asyncio
        t = np.linspace(0, 8 * math.pi, 200)
        signal = np.sin(t).tolist()
        window = _make_window({"sensor_b": signal})
        # Very high threshold — clean sine wave should not exceed it
        detector = DampDetector(sensor_ids=["sensor_b"], window_length=20, threshold=999.0)
        events = asyncio.run(detector.evaluate(window))
        assert events == []

    def test_discord_timestamp_within_window(self):
        import asyncio
        signal = _sine_with_discord(n=200, discord_idx=150)
        window = _make_window({"sensor_c": signal})
        detector = DampDetector(sensor_ids=["sensor_c"], window_length=20, threshold=1.0)
        events = asyncio.run(detector.evaluate(window))
        assert len(events) == 1
        assert window.start <= events[0].timestamp <= window.end

    def test_details_populated(self):
        import asyncio
        signal = _sine_with_discord(n=200)
        window = _make_window({"sensor_d": signal})
        detector = DampDetector(sensor_ids=["sensor_d"], window_length=20, threshold=1.0)
        events = asyncio.run(detector.evaluate(window))
        assert len(events) == 1
        d = events[0].details
        assert "discord_index" in d
        assert "window_length" in d
        assert d["window_length"] == 20


class TestDampDetectorEdgeCases:
    def test_short_window_returns_empty(self):
        """Fewer than 2*window_length samples must not crash — returns []."""
        import asyncio
        signal = [1.0, 2.0, 3.0]  # only 3 samples; window_length=10 needs 20
        window = _make_window({"sensor_e": signal})
        detector = DampDetector(sensor_ids=["sensor_e"], window_length=10, threshold=0.1)
        events = asyncio.run(detector.evaluate(window))
        assert events == []

    def test_missing_sensor_skipped(self):
        """A sensor_id not present in frames must be silently skipped."""
        import asyncio
        signal = np.sin(np.linspace(0, 4 * math.pi, 200)).tolist()
        window = _make_window({"sensor_f": signal})
        detector = DampDetector(
            sensor_ids=["sensor_f", "sensor_does_not_exist"],
            window_length=20,
            threshold=999.0,
        )
        events = asyncio.run(detector.evaluate(window))
        # No crash; no events (threshold is very high)
        assert isinstance(events, list)

    def test_multiple_sensors_independent(self):
        """Two sensors with discords both get flagged independently."""
        import asyncio
        sig_a = _sine_with_discord(n=200, discord_idx=100, discord_mag=10.0)
        sig_b = _sine_with_discord(n=200, discord_idx=160, discord_mag=10.0)
        window = _make_window({"s_a": sig_a, "s_b": sig_b})
        detector = DampDetector(sensor_ids=["s_a", "s_b"], window_length=20, threshold=1.0)
        events = asyncio.run(detector.evaluate(window))
        flagged = {e.affected_sensors[0] for e in events}
        assert "s_a" in flagged
        assert "s_b" in flagged
