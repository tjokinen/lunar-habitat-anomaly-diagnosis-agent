"""Tests for core interface Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from selene.core.interfaces import (
    AnomalyEvent,
    AnomalyGroundTruth,
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
    TelemetryWindow,
)

NOW = datetime(2020, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
LATER = datetime(2020, 3, 1, 13, 0, 0, tzinfo=timezone.utc)


def make_reading(sensor_id: str = "tcs_temp_01", value: float = 22.5) -> SensorReading:
    return SensorReading(sensor_id=sensor_id, timestamp=NOW, value=value, unit="degC")


def make_frame(readings: dict | None = None) -> TelemetryFrame:
    if readings is None:
        readings = {"tcs_temp_01": make_reading()}
    return TelemetryFrame(timestamp=NOW, readings=readings)


class TestSensorReading:
    def test_valid(self):
        r = make_reading()
        assert r.sensor_id == "tcs_temp_01"
        assert r.value == 22.5
        assert r.unit == "degC"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            SensorReading.model_validate({"sensor_id": "s", "timestamp": NOW, "unit": "degC"})

    def test_wrong_type_value(self):
        with pytest.raises(ValidationError):
            SensorReading.model_validate(
                {"sensor_id": "s", "timestamp": NOW, "value": "not_a_float", "unit": "degC"}
            )


class TestTelemetryFrame:
    def test_valid(self):
        frame = make_frame()
        assert frame.timestamp == NOW
        assert "tcs_temp_01" in frame.readings

    def test_default_metadata(self):
        frame = make_frame()
        assert frame.metadata == {}

    def test_custom_metadata(self):
        frame = TelemetryFrame(timestamp=NOW, readings={}, metadata={"source": "test"})
        assert frame.metadata["source"] == "test"

    def test_missing_timestamp(self):
        with pytest.raises(ValidationError):
            TelemetryFrame.model_validate({"readings": {}})


class TestSensorMetadata:
    def test_valid(self):
        meta = SensorMetadata(
            sensors={"tcs_temp_01": {"unit": "degC", "subsystem": "TCS"}},
            subsystems=["TCS"],
            sampling_rate_seconds=60.0,
        )
        assert meta.sampling_rate_seconds == 60.0
        assert "TCS" in meta.subsystems

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            SensorMetadata.model_validate({"sensors": {}, "subsystems": []})


class TestAnomalyGroundTruth:
    def test_valid(self):
        gt = AnomalyGroundTruth(
            scenario_id="test_scenario",
            start_time=NOW,
            end_time=LATER,
            affected_sensors=["tcs_temp_01"],
            description="A test anomaly",
        )
        assert gt.scenario_id == "test_scenario"
        assert gt.affected_sensors == ["tcs_temp_01"]

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            AnomalyGroundTruth.model_validate(
                {
                    "scenario_id": "s",
                    "start_time": NOW,
                    "end_time": LATER,
                    "affected_sensors": [],
                }
            )


class TestTelemetryWindow:
    def test_valid(self):
        frames = [make_frame()]
        window = TelemetryWindow(frames=frames, start=NOW, end=LATER)
        assert len(window.frames) == 1
        assert window.start == NOW

    def test_empty_frames(self):
        window = TelemetryWindow(frames=[], start=NOW, end=LATER)
        assert window.frames == []


class TestAnomalyEvent:
    def test_valid(self):
        event = AnomalyEvent(
            detector_name="damp",
            timestamp=NOW,
            affected_sensors=["tcs_temp_01"],
            score=0.87,
        )
        assert event.detector_name == "damp"
        assert event.score == 0.87
        assert event.details == {}

    def test_with_details(self):
        event = AnomalyEvent(
            detector_name="threshold",
            timestamp=NOW,
            affected_sensors=["tcs_temp_01"],
            score=1.5,
            details={"min": 0.0, "max": 100.0, "actual": 105.0},
        )
        assert event.details["actual"] == 105.0

    def test_missing_score(self):
        with pytest.raises(ValidationError):
            AnomalyEvent.model_validate(
                {
                    "detector_name": "damp",
                    "timestamp": NOW,
                    "affected_sensors": [],
                }
            )
