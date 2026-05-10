"""StepChangeAnomaly — adds a constant offset to a sensor between two times."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from selene.core.interfaces import AnomalyGroundTruth, TelemetryFrame
from selene.scenarios.registry import register_module


class StepChangeConfig(BaseModel):
    name: str
    description: str
    affected_sensors: list[str]
    start_time: datetime
    end_time: datetime
    offset: float
    unit: str = ""


@register_module("step_change")
class StepChangeAnomaly:
    """Adds a fixed offset to one or more sensors within a time window."""

    def __init__(self, **kwargs) -> None:
        cfg = StepChangeConfig(**kwargs)
        self.name: str = cfg.name
        self.description: str = cfg.description
        self.affected_sensors: list[str] = cfg.affected_sensors
        self._start = cfg.start_time
        self._end = cfg.end_time
        self._offset = cfg.offset
        self._unit = cfg.unit

    def applies_at(self, t: datetime) -> bool:
        t_naive = t.replace(tzinfo=None) if t.tzinfo else t
        start = self._start.replace(tzinfo=None) if self._start.tzinfo else self._start
        end = self._end.replace(tzinfo=None) if self._end.tzinfo else self._end
        return start <= t_naive <= end

    def transform(self, frame: TelemetryFrame) -> TelemetryFrame:
        if not self.applies_at(frame.timestamp):
            return frame

        new_readings = dict(frame.readings)
        for sensor_id in self.affected_sensors:
            if sensor_id not in new_readings:
                continue
            original = new_readings[sensor_id]
            new_readings[sensor_id] = original.model_copy(
                update={"value": original.value + self._offset}
            )

        return frame.model_copy(update={"readings": new_readings})

    def get_ground_truth(self) -> AnomalyGroundTruth:
        return AnomalyGroundTruth(
            scenario_id=self.name,
            start_time=self._start,
            end_time=self._end,
            affected_sensors=self.affected_sensors,
            description=self.description,
        )
