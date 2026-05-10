"""Tests for MdiDetector.

Headline test: synthetic two-sensor stream where the marginals stay identical
across baseline and anomaly, but the joint covariance changes — a univariate
detector cannot flag this, MDI must.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from selene.core.interfaces import SensorReading, TelemetryFrame, TelemetryWindow
from selene.detection.mdi import MdiDetector

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


def _correlated_then_decoupled(
    n_baseline: int = 200,
    n_anomaly: int = 100,
    n_post: int = 50,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Two sensors anti-correlated in baseline, independent during the anomaly,
    anti-correlated again post-anomaly. Both marginals stay ≈ N(0, 1) throughout.
    """
    rng = np.random.default_rng(seed)

    x_base = rng.standard_normal(n_baseline)
    y_base = -x_base + 0.05 * rng.standard_normal(n_baseline)

    x_anom = rng.standard_normal(n_anomaly)
    y_anom = rng.standard_normal(n_anomaly)  # independent of x_anom

    x_post = rng.standard_normal(n_post)
    y_post = -x_post + 0.05 * rng.standard_normal(n_post)

    x = np.concatenate([x_base, x_anom, x_post]).tolist()
    y = np.concatenate([y_base, y_anom, y_post]).tolist()
    return x, y


class TestMdiDetectorMultivariate:
    def test_flags_decoupling_no_marginal_shift(self):
        """The headline case: two sensors decouple but each marginal stays N(0,1)."""
        x, y = _correlated_then_decoupled(n_baseline=200, n_anomaly=100, n_post=50)
        window = _make_window({"x": x, "y": y})

        detector = MdiDetector(
            sensor_ids=["x", "y"],
            baseline_window_size=200,
            detection_window_size=40,
            threshold=2.0,
        )
        events = asyncio.run(detector.evaluate(window))

        assert len(events) >= 1, "MDI should flag the decoupling interval"
        assert events[0].detector_name == "mdi"
        assert set(events[0].affected_sensors) == {"x", "y"}
        assert events[0].score > 2.0

        # The flagged interval should overlap the injected anomaly region
        # (samples 200..299 in the concatenated series).
        interval_start = datetime.fromisoformat(events[0].details["interval_start"])
        interval_end = datetime.fromisoformat(events[0].details["interval_end"])
        anomaly_start = _T0 + 200 * _DT
        anomaly_end = _T0 + 300 * _DT
        # Overlap check (any intersection)
        assert interval_start < anomaly_end and interval_end > anomaly_start

    def test_marginals_alone_would_not_flag(self):
        """Sanity check on the test design: each sensor's marginal in the anomaly
        region is statistically indistinguishable from baseline (no univariate
        detector tuned to either sensor alone could reliably catch this)."""
        x, y = _correlated_then_decoupled(n_baseline=200, n_anomaly=100, n_post=50, seed=42)
        x = np.array(x)
        y = np.array(y)
        for s, label in [(x, "x"), (y, "y")]:
            base_mean = s[:200].mean()
            anom_mean = s[200:300].mean()
            base_std = s[:200].std()
            anom_std = s[200:300].std()
            # Means within ~2 standard errors of zero, stds within ~30% of each other
            assert abs(base_mean - anom_mean) < 0.5, f"{label}: marginal mean shift too large"
            assert abs(base_std - anom_std) / base_std < 0.3, f"{label}: marginal std shift too large"


class TestMdiDetectorBasics:
    def test_no_event_on_stationary_baseline(self):
        """If the whole stream is one stationary distribution, no events fire."""
        rng = np.random.default_rng(0)
        n = 400
        x = rng.standard_normal(n)
        y = -x + 0.05 * rng.standard_normal(n)
        window = _make_window({"x": x.tolist(), "y": y.tolist()})

        detector = MdiDetector(
            sensor_ids=["x", "y"],
            baseline_window_size=150,
            detection_window_size=40,
            threshold=2.0,
        )
        events = asyncio.run(detector.evaluate(window))
        assert events == []

    def test_short_input_returns_empty(self):
        """Fewer than baseline + detection samples → no crash, returns []."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.5, 0.4, 0.3, 0.2, 0.1]
        window = _make_window({"x": x, "y": y})

        detector = MdiDetector(
            sensor_ids=["x", "y"],
            baseline_window_size=10,
            detection_window_size=5,
            threshold=0.1,
        )
        events = asyncio.run(detector.evaluate(window))
        assert events == []

    def test_missing_sensor_drops_frame(self):
        """Frames missing a requested sensor are dropped from the matrix; if too
        many drop out the detector simply emits nothing rather than crashing."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50).tolist()
        # Build a stream that only has sensor x — sensor y is requested but absent
        window = _make_window({"x": x})
        detector = MdiDetector(
            sensor_ids=["x", "y"],
            baseline_window_size=20,
            detection_window_size=10,
            threshold=0.1,
        )
        events = asyncio.run(detector.evaluate(window))
        assert events == []  # Every frame drops, no usable matrix

    def test_event_payload_shape(self):
        """An emitted event carries the expected keys and timestamp range."""
        x, y = _correlated_then_decoupled(n_baseline=200, n_anomaly=100, n_post=50)
        window = _make_window({"x": x, "y": y})
        detector = MdiDetector(
            sensor_ids=["x", "y"],
            baseline_window_size=200,
            detection_window_size=40,
            threshold=2.0,
        )
        events = asyncio.run(detector.evaluate(window))
        assert events
        ev = events[0]
        for key in ("interval_start", "interval_end", "baseline_window_size",
                    "detection_window_size", "n_above_threshold", "kl_peak"):
            assert key in ev.details
        assert ev.details["baseline_window_size"] == 200
        assert ev.details["detection_window_size"] == 40
        assert window.start <= ev.timestamp <= window.end


class TestMdiDetectorConstructorValidation:
    def test_rejects_baseline_too_small(self):
        with pytest.raises(ValueError, match="baseline"):
            MdiDetector(sensor_ids=["x"], baseline_window_size=1,
                        detection_window_size=10, threshold=1.0)

    def test_rejects_detection_too_small(self):
        with pytest.raises(ValueError, match="detection"):
            MdiDetector(sensor_ids=["x"], baseline_window_size=10,
                        detection_window_size=1, threshold=1.0)

    def test_rejects_empty_sensor_list(self):
        with pytest.raises(ValueError, match="sensor_ids"):
            MdiDetector(sensor_ids=[], baseline_window_size=10,
                        detection_window_size=10, threshold=1.0)


class TestMdiDetectorUnivariateFallback:
    """MDI is multivariate-by-design but should also work with a single sensor."""

    def test_single_sensor_mean_shift(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([
            rng.standard_normal(200),
            rng.standard_normal(80) + 5.0,  # mean shift
            rng.standard_normal(50),
        ]).tolist()
        window = _make_window({"x": x})
        detector = MdiDetector(
            sensor_ids=["x"],
            baseline_window_size=200,
            detection_window_size=30,
            threshold=2.0,
        )
        events = asyncio.run(detector.evaluate(window))
        assert events
        assert events[0].score > 2.0
