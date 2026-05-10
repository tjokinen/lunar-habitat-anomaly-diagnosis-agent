"""ThresholdDetector — rule-based fallback for known sensor operational ranges."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from selene.core.interfaces import AnomalyEvent, SensorMetadata, TelemetryWindow

logger = logging.getLogger(__name__)


class ThresholdDetector:
    """Flags sensor values that fall outside their configured operational range.

    Score = how far outside the range the value is, normalised by range width.
    A value exactly at the boundary scores 0; one range-width outside scores 1.

    Args:
        ranges: Mapping of sensor_id -> (min, max).  Sensors absent from this
            dict are silently skipped.
    """

    name = "threshold"

    def __init__(self, ranges: dict[str, tuple[float, float]]) -> None:
        self._ranges = ranges

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        metadata: SensorMetadata | None = None,
    ) -> "ThresholdDetector":
        """Build a ThresholdDetector from a YAML range config.

        The YAML must have a ``type_defaults`` section (sensor_type -> [min, max])
        and an optional ``sensor_overrides`` section (sensor_id -> [min, max]).

        If ``metadata`` is provided, every sensor in the metadata is resolved:
        first checking ``sensor_overrides``, then falling back to the
        ``type_defaults`` entry for that sensor's ``sensor_type``.  Sensors
        with no applicable default are skipped with a debug log.

        If ``metadata`` is None, only the ``sensor_overrides`` entries are used.
        """
        path = Path(config_path)
        with path.open() as f:
            cfg: dict = yaml.safe_load(f)

        type_defaults: dict[str, list[float]] = cfg.get("type_defaults", {})
        sensor_overrides: dict[str, list[float]] = cfg.get("sensor_overrides", {})

        ranges: dict[str, tuple[float, float]] = {}

        if metadata is not None:
            for sensor_id, info in metadata.sensors.items():
                if sensor_id in sensor_overrides:
                    lo, hi = sensor_overrides[sensor_id]
                else:
                    stype = info.get("sensor_type", "")
                    if stype not in type_defaults:
                        logger.debug(
                            "ThresholdDetector: no range for sensor %s (type=%r)", sensor_id, stype
                        )
                        continue
                    lo, hi = type_defaults[stype]
                ranges[sensor_id] = (float(lo), float(hi))
        else:
            for sensor_id, (lo, hi) in sensor_overrides.items():
                ranges[sensor_id] = (float(lo), float(hi))

        return cls(ranges)

    # ------------------------------------------------------------------
    # AnomalyDetector protocol
    # ------------------------------------------------------------------

    async def evaluate(self, window: TelemetryWindow) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []

        for frame in window.frames:
            for sensor_id, (lo, hi) in self._ranges.items():
                reading = frame.readings.get(sensor_id)
                if reading is None:
                    continue

                value = reading.value
                range_width = hi - lo if hi != lo else 1.0

                if value < lo:
                    score = (lo - value) / range_width
                elif value > hi:
                    score = (value - hi) / range_width
                else:
                    continue

                events.append(
                    AnomalyEvent(
                        detector_name=self.name,
                        timestamp=frame.timestamp,
                        affected_sensors=[sensor_id],
                        score=score,
                        details={
                            "value": value,
                            "range_min": lo,
                            "range_max": hi,
                            "unit": reading.unit,
                        },
                    )
                )

        return events
