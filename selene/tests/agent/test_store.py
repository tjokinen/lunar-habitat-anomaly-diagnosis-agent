"""Tests for selene.agent.store.TelemetryStore."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from selene.agent.store import TelemetryStore
from selene.core.interfaces import (
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
)


_T0 = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
_DT = timedelta(minutes=5)


def _frame(i: int, values: dict[str, tuple[float, str]]) -> TelemetryFrame:
    """Build a TelemetryFrame at time _T0 + i*_DT from {sid: (value, unit)}."""
    ts = _T0 + i * _DT
    readings = {
        sid: SensorReading(sensor_id=sid, timestamp=ts, value=v, unit=u)
        for sid, (v, u) in values.items()
    }
    return TelemetryFrame(timestamp=ts, readings=readings)


class TestIngestAndHistory:
    def test_history_empty_for_unknown_sensor(self):
        store = TelemetryStore()
        ts = store.history("nope", _T0, _T0 + 10 * _DT)
        assert ts.points == []
        assert ts.sensor_id == "nope"
        assert ts.unit == ""

    def test_ingest_appends_in_order(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        for i in range(5):
            store.ingest(_frame(i, {"a": (float(i), "ppm")}))
        ts = store.history("a", _T0, _T0 + 5 * _DT)
        assert [p.value for p in ts.points] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert ts.unit == "ppm"

    def test_history_respects_inclusive_bounds(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        for i in range(5):
            store.ingest(_frame(i, {"a": (float(i), "")}))
        ts = store.history("a", _T0 + _DT, _T0 + 3 * _DT)
        assert [p.value for p in ts.points] == [1.0, 2.0, 3.0]


class TestRetentionEviction:
    def test_evicts_entries_older_than_retention(self):
        store = TelemetryStore(retention=2 * _DT)
        for i in range(5):
            store.ingest(_frame(i, {"a": (float(i), "")}))
        # After ingesting frame 4 (t=20min), retention 10min keeps frames at
        # t in [10min, 20min] → values [2.0, 3.0, 4.0]
        ts = store.history("a", _T0, _T0 + 10 * _DT)
        assert [p.value for p in ts.points] == [2.0, 3.0, 4.0]

    def test_eviction_uses_latest_frame_timestamp(self):
        store = TelemetryStore(retention=_DT)
        # Ingest a single frame far in the past — it stays (it's the only one)
        store.ingest(_frame(0, {"a": (1.0, "")}))
        # Now ingest a frame 1h later; only frames within `retention` of *that*
        # frame survive. So frame 0 is evicted.
        store.ingest(
            TelemetryFrame(
                timestamp=_T0 + timedelta(hours=1),
                readings={
                    "a": SensorReading(
                        sensor_id="a",
                        timestamp=_T0 + timedelta(hours=1),
                        value=99.0,
                        unit="",
                    )
                },
            )
        )
        ts = store.history("a", _T0, _T0 + timedelta(hours=2))
        assert [p.value for p in ts.points] == [99.0]


class TestLatest:
    def test_latest_returns_most_recent_per_sensor(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        for i in range(3):
            store.ingest(_frame(i, {"a": (float(i), "x"), "b": (float(i * 10), "y")}))
        latest = store.latest(["a", "b"])
        assert latest["a"].value == 2.0
        assert latest["b"].value == 20.0
        assert latest["a"].unit == "x"

    def test_latest_omits_unseen_sensors(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        store.ingest(_frame(0, {"a": (1.0, "")}))
        latest = store.latest(["a", "missing"])
        assert "missing" not in latest
        assert latest["a"].value == 1.0

    def test_latest_empty_input_returns_empty(self):
        store = TelemetryStore()
        assert store.latest([]) == {}


class TestSubsystemState:
    def _metadata(self) -> SensorMetadata:
        return SensorMetadata(
            sensors={
                "a": {"subsystem": "atmosphere_management_system", "unit": "ppm"},
                "b": {"subsystem": "atmosphere_management_system", "unit": "%"},
                "c": {"subsystem": "thermal_control_system", "unit": "C"},
            },
            subsystems=["atmosphere_management_system", "thermal_control_system"],
            sampling_rate_seconds=300.0,
        )

    def test_returns_only_requested_subsystem(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        store.ingest(
            _frame(0, {"a": (410.0, "ppm"), "b": (60.0, "%"), "c": (22.0, "C")})
        )
        snap = store.subsystem_state("atmosphere_management_system", self._metadata())
        assert set(snap.readings) == {"a", "b"}
        assert snap.readings["a"] == 410.0
        assert snap.units["a"] == "ppm"

    def test_anchors_at_most_recent_timestamp(self):
        store = TelemetryStore(retention=timedelta(hours=24))
        store.ingest(_frame(0, {"a": (1.0, "ppm")}))
        store.ingest(_frame(2, {"b": (2.0, "%")}))
        snap = store.subsystem_state("atmosphere_management_system", self._metadata())
        assert snap.timestamp == _T0 + 2 * _DT

    def test_empty_subsystem_returns_default_timestamp(self):
        store = TelemetryStore()
        snap = store.subsystem_state("atmosphere_management_system", self._metadata())
        assert snap.readings == {}
        assert snap.units == {}
        assert snap.timestamp.tzinfo is not None  # Always a tz-aware sentinel


class TestDeterminism:
    def test_same_inputs_yield_same_history(self):
        def build() -> TelemetryStore:
            store = TelemetryStore(retention=timedelta(hours=24))
            for i in range(5):
                store.ingest(_frame(i, {"a": (float(i), "")}))
            return store

        ts1 = build().history("a", _T0, _T0 + 5 * _DT)
        ts2 = build().history("a", _T0, _T0 + 5 * _DT)
        assert ts1 == ts2
