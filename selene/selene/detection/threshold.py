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

    Two filters keep noise out of the pipeline:

    - ``min_score`` (default 1.0): only emit when the value is at least one
      range-width past the boundary.  Marginal excursions that don't clear
      this bar are ignored.
    - ``consecutive_frames`` (default 3): only emit when the *last* N frames
      of the window are *all* out of range.  Single-frame transients are
      ignored.  At one event-per-evaluate, we never spam the queue from a
      stuck-at-zero sensor.

    Args:
        ranges: Mapping of sensor_id -> (min, max).  Sensors absent from this
            dict are silently skipped.
        min_score: Minimum normalised score required to emit. Default 1.0.
        consecutive_frames: Required number of consecutive out-of-range frames
            at the tail of the window.  Default 3 (≈15 min at 5-min cadence).
    """

    name = "threshold"

    def __init__(
        self,
        ranges: dict[str, tuple[float, float]],
        *,
        min_score: float = 1.0,
        consecutive_frames: int = 3,
    ) -> None:
        self._ranges = ranges
        self._min_score = min_score
        self._consecutive_frames = max(1, consecutive_frames)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        metadata: SensorMetadata | None = None,
        *,
        min_score: float = 1.0,
        consecutive_frames: int = 3,
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

        return cls(ranges, min_score=min_score, consecutive_frames=consecutive_frames)

    # ------------------------------------------------------------------
    # AnomalyDetector protocol
    # ------------------------------------------------------------------

    async def evaluate(self, window: TelemetryWindow) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        n = self._consecutive_frames
        if len(window.frames) < n:
            return events

        recent = window.frames[-n:]
        latest = recent[-1]

        for sensor_id, (lo, hi) in self._ranges.items():
            range_width = hi - lo if hi != lo else 1.0

            # Require all N tail frames to be present and out-of-range.
            scores: list[float] = []
            for f in recent:
                reading = f.readings.get(sensor_id)
                if reading is None:
                    scores = []
                    break
                v = reading.value
                if v < lo:
                    scores.append((lo - v) / range_width)
                elif v > hi:
                    scores.append((v - hi) / range_width)
                else:
                    scores = []
                    break
            if not scores:
                continue

            score = max(scores)
            if score < self._min_score:
                continue

            latest_reading = latest.readings[sensor_id]
            events.append(
                AnomalyEvent(
                    detector_name=self.name,
                    timestamp=latest.timestamp,
                    affected_sensors=[sensor_id],
                    score=score,
                    details={
                        "value": latest_reading.value,
                        "range_min": lo,
                        "range_max": hi,
                        "unit": latest_reading.unit,
                    },
                )
            )

        return events
