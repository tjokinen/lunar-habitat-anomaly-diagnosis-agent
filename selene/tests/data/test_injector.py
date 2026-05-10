"""Tests for ScenarioInjector."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from selene.core.interfaces import (
    AnomalyGroundTruth,
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
)
from selene.data.injector import ScenarioInjector

T0 = datetime(2020, 6, 1, 0, 5, 0, tzinfo=timezone.utc)
T1 = datetime(2020, 6, 1, 1, 5, 0, tzinfo=timezone.utc)
T2 = datetime(2020, 6, 1, 2, 5, 0, tzinfo=timezone.utc)
T3 = datetime(2020, 6, 1, 3, 5, 0, tzinfo=timezone.utc)
T4 = datetime(2020, 6, 1, 4, 5, 0, tzinfo=timezone.utc)  # end of anomaly window


def make_frame(ts: datetime, value: float = 20.0) -> TelemetryFrame:
    reading = SensorReading(
        sensor_id="tcs/temp-ams_in",
        timestamp=ts,
        value=value,
        unit="degrees celsius",
    )
    return TelemetryFrame(timestamp=ts, readings={"tcs/temp-ams_in": reading})


class StubSource:
    """Minimal TelemetrySource that yields a fixed list of frames."""

    def __init__(self, frames: list[TelemetryFrame]) -> None:
        self._frames = frames
        self._meta = SensorMetadata(
            sensors={"tcs/temp-ams_in": {"unit": "degrees celsius", "subsystem": "TCS"}},
            subsystems=["TCS"],
            sampling_rate_seconds=300.0,
        )

    async def stream(self):
        for f in self._frames:
            yield f

    def get_metadata(self) -> SensorMetadata:
        return self._meta


class NoOpModule:
    """AnomalyModule that never transforms anything."""

    name = "noop"
    description = "no-op"
    affected_sensors: list[str] = []

    def applies_at(self, t: datetime) -> bool:
        return False

    def transform(self, frame: TelemetryFrame) -> TelemetryFrame:
        return frame

    def get_ground_truth(self) -> AnomalyGroundTruth:
        return AnomalyGroundTruth(
            scenario_id="noop",
            start_time=T0,
            end_time=T4,
            affected_sensors=[],
            description="no-op",
        )


class StepModule:
    """Simple step-change module for testing (no YAML involved)."""

    name = "step"
    description = "step change on tcs/temp-ams_in"
    affected_sensors = ["tcs/temp-ams_in"]

    def __init__(self, start: datetime, end: datetime, offset: float = 5.0) -> None:
        self._start = start
        self._end = end
        self._offset = offset

    def applies_at(self, t: datetime) -> bool:
        t_naive = t.replace(tzinfo=None) if t.tzinfo else t
        s = self._start.replace(tzinfo=None) if self._start.tzinfo else self._start
        e = self._end.replace(tzinfo=None) if self._end.tzinfo else self._end
        return s <= t_naive <= e

    def transform(self, frame: TelemetryFrame) -> TelemetryFrame:
        if not self.applies_at(frame.timestamp):
            return frame
        new_readings = dict(frame.readings)
        orig = new_readings["tcs/temp-ams_in"]
        new_readings["tcs/temp-ams_in"] = orig.model_copy(
            update={"value": orig.value + self._offset}
        )
        return frame.model_copy(update={"readings": new_readings})

    def get_ground_truth(self) -> AnomalyGroundTruth:
        return AnomalyGroundTruth(
            scenario_id="step",
            start_time=self._start,
            end_time=self._end,
            affected_sensors=self.affected_sensors,
            description="step change",
        )


async def collect(injector: ScenarioInjector) -> list[TelemetryFrame]:
    return [f async for f in injector.stream()]


class TestScenarioInjectorNoOp:
    def test_yields_identical_frames(self):
        frames = [make_frame(T0), make_frame(T1), make_frame(T2)]
        source = StubSource(frames)
        injector = ScenarioInjector(source, [NoOpModule()])
        result = asyncio.run(collect(injector))
        assert len(result) == 3
        for original, transformed in zip(frames, result):
            assert transformed.readings["tcs/temp-ams_in"].value == original.readings["tcs/temp-ams_in"].value

    def test_get_metadata_delegates(self):
        source = StubSource([])
        injector = ScenarioInjector(source, [])
        assert injector.get_metadata() is source.get_metadata()


class TestScenarioInjectorStepChange:
    def _make_injector(self) -> tuple[ScenarioInjector, list[TelemetryFrame]]:
        frames = [make_frame(T0), make_frame(T1), make_frame(T2), make_frame(T3), make_frame(T4)]
        source = StubSource(frames)
        module = StepModule(start=T2, end=T4, offset=5.0)
        injector = ScenarioInjector(source, [module])
        return injector, frames

    def test_pre_window_frames_unmodified(self):
        injector, _ = self._make_injector()
        result = asyncio.run(collect(injector))
        # T0 and T1 are before the anomaly window
        assert result[0].readings["tcs/temp-ams_in"].value == pytest.approx(20.0)
        assert result[1].readings["tcs/temp-ams_in"].value == pytest.approx(20.0)

    def test_in_window_frames_offset(self):
        injector, _ = self._make_injector()
        result = asyncio.run(collect(injector))
        # T2, T3, T4 are inside the window
        assert result[2].readings["tcs/temp-ams_in"].value == pytest.approx(25.0)
        assert result[3].readings["tcs/temp-ams_in"].value == pytest.approx(25.0)
        assert result[4].readings["tcs/temp-ams_in"].value == pytest.approx(25.0)

    def test_original_frames_not_mutated(self):
        """transform() must not mutate the original frame."""
        frames = [make_frame(T2)]
        source = StubSource(frames)
        module = StepModule(start=T2, end=T4, offset=5.0)
        injector = ScenarioInjector(source, [module])
        asyncio.run(collect(injector))
        assert frames[0].readings["tcs/temp-ams_in"].value == pytest.approx(20.0)

    def test_module_order_is_deterministic(self):
        """Two additive modules applied in order should stack."""
        frames = [make_frame(T2)]
        source = StubSource(frames)
        m1 = StepModule(start=T2, end=T4, offset=3.0)
        m2 = StepModule(start=T2, end=T4, offset=2.0)
        injector = ScenarioInjector(source, [m1, m2])
        result = asyncio.run(collect(injector))
        assert result[0].readings["tcs/temp-ams_in"].value == pytest.approx(25.0)
