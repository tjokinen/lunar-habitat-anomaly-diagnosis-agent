"""Core interfaces and shared types for the Selene pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel


class SensorReading(BaseModel):
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str


class TelemetryFrame(BaseModel):
    timestamp: datetime
    readings: dict[str, SensorReading]  # keyed by sensor_id
    metadata: dict = {}


class SensorMetadata(BaseModel):
    sensors: dict[str, dict]  # sensor_id -> {unit, subsystem, range, etc}
    subsystems: list[str]
    sampling_rate_seconds: float


class AnomalyGroundTruth(BaseModel):
    scenario_id: str
    start_time: datetime
    end_time: datetime
    affected_sensors: list[str]
    description: str


class TelemetryWindow(BaseModel):
    frames: list[TelemetryFrame]
    start: datetime
    end: datetime


class AnomalyEvent(BaseModel):
    detector_name: str
    timestamp: datetime
    affected_sensors: list[str]
    score: float  # detector-specific anomaly score
    details: dict = {}


@runtime_checkable
class TelemetrySource(Protocol):
    async def stream(self) -> AsyncIterator[TelemetryFrame]: ...
    def get_metadata(self) -> SensorMetadata: ...


@runtime_checkable
class AnomalyModule(Protocol):
    name: str
    description: str
    affected_sensors: list[str]

    def applies_at(self, t: datetime) -> bool: ...
    def transform(self, frame: TelemetryFrame) -> TelemetryFrame: ...
    def get_ground_truth(self) -> AnomalyGroundTruth: ...


@runtime_checkable
class AnomalyDetector(Protocol):
    name: str

    async def evaluate(self, window: TelemetryWindow) -> list[AnomalyEvent]: ...
