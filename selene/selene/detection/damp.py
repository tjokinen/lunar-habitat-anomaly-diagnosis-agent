"""DAMP detector — Discord-Aware Matrix Profile via stumpy."""

from __future__ import annotations

import logging

import numpy as np
import stumpy

from selene.core.interfaces import AnomalyEvent, TelemetryWindow

logger = logging.getLogger(__name__)

# stumpy requires at least 2 * window_length data points to compute a matrix profile
_MIN_SAMPLES_FACTOR = 2


class DampDetector:
    """Detects discords (anomalous subsequences) using the matrix profile.

    Runs stumpy.stump on each requested sensor independently and flags
    sensors whose discord score (max matrix-profile value) exceeds ``threshold``.

    Args:
        sensor_ids: Sensors to evaluate.
        window_length: Subsequence length (in samples) for the matrix profile.
        threshold: Discord score above which an AnomalyEvent is raised.
    """

    name = "damp"

    def __init__(
        self,
        sensor_ids: list[str],
        window_length: int,
        threshold: float,
    ) -> None:
        self._sensor_ids = sensor_ids
        self._window_length = window_length
        self._threshold = threshold

    async def evaluate(self, window: TelemetryWindow) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []

        for sensor_id in self._sensor_ids:
            values = self._extract_values(window, sensor_id)
            if values is None:
                continue

            min_samples = _MIN_SAMPLES_FACTOR * self._window_length
            if len(values) < min_samples:
                logger.debug(
                    "DampDetector: sensor %s has %d samples, need %d — skipping",
                    sensor_id,
                    len(values),
                    min_samples,
                )
                continue

            event = self._evaluate_sensor(window, sensor_id, values)
            if event is not None:
                events.append(event)

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_values(
        self, window: TelemetryWindow, sensor_id: str
    ) -> np.ndarray | None:
        vals: list[float] = []
        for frame in window.frames:
            reading = frame.readings.get(sensor_id)
            if reading is None or np.isnan(reading.value):
                continue
            vals.append(reading.value)

        if not vals:
            return None
        return np.array(vals, dtype=np.float64)

    def _evaluate_sensor(
        self,
        window: TelemetryWindow,
        sensor_id: str,
        values: np.ndarray,
    ) -> AnomalyEvent | None:
        try:
            mp = stumpy.stump(values, m=self._window_length)
        except Exception as exc:
            logger.warning("DampDetector: stumpy.stump failed for %s: %s", sensor_id, exc)
            return None

        # Matrix profile column 0 = distances; discord is at the maximum
        mp_distances: np.ndarray = mp[:, 0].astype(float)
        discord_idx = int(np.argmax(mp_distances))
        discord_score = float(mp_distances[discord_idx])

        if discord_score <= self._threshold:
            return None

        # Map discord index back to the window timestamp
        discord_frame_idx = min(discord_idx, len(window.frames) - 1)
        discord_ts = window.frames[discord_frame_idx].timestamp

        return AnomalyEvent(
            detector_name=self.name,
            timestamp=discord_ts,
            affected_sensors=[sensor_id],
            score=discord_score,
            details={
                "discord_index": discord_idx,
                "window_length": self._window_length,
                "mp_mean": float(np.mean(mp_distances)),
                "mp_max": discord_score,
            },
        )
