"""In-memory telemetry buffer with bounded retention.

Backed by per-sensor deques keyed by ``sensor_id``. On each ingest, entries
older than ``retention`` (relative to the incoming frame's timestamp) are
evicted from every touched sensor's deque.

The store is the agent's read view over recent telemetry — tools query it via
``query_sensor_history``, ``fetch_subsystem_state``, and ``correlate_signals``
in ``selene/agent/tools.py``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from selene.agent.types import SubsystemSnapshot, TimeSeries, TimeSeriesPoint
from selene.core.interfaces import SensorMetadata, SensorReading, TelemetryFrame

# Alias map: long-form subsystem names (advertised in the LLM tool schema /
# system prompt) → short codes that actually appear in EDEN ISS metadata.
# Both directions are accepted by ``subsystem_state`` so the agent can pass
# either form. Match is case-insensitive on the key.
_SUBSYSTEM_ALIASES: dict[str, str] = {
    "thermal_control_system":      "tcs",
    "tcs":                         "tcs",
    "atmosphere_management_system": "ams",
    "ams":                         "ams",
    "nutrient_delivery_system":    "nds",
    "nds":                         "nds",
    "illumination_control_system": "ics",
    "ics":                         "ics",
}


def _normalise_subsystem(name: str) -> str:
    """Map either form (long or short, any case) to a canonical lowercase code."""
    return _SUBSYSTEM_ALIASES.get(name.lower(), name.lower())


class TelemetryStore:
    """Bounded ring buffer of recent ``SensorReading``s, keyed by sensor_id.

    Args:
        retention: How far back to keep readings, measured from the most recent
            ingested frame's timestamp. Defaults to one hour.
    """

    def __init__(self, retention: timedelta = timedelta(hours=1)) -> None:
        self._retention = retention
        self._readings: dict[str, deque[SensorReading]] = defaultdict(deque)

    def ingest(self, frame: TelemetryFrame) -> None:
        """Append every reading in ``frame`` to its sensor's deque, then evict
        entries older than ``retention`` from those same deques."""
        cutoff = frame.timestamp - self._retention
        for sensor_id, reading in frame.readings.items():
            buf = self._readings[sensor_id]
            buf.append(reading)
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()

    def history(self, sensor_id: str, start: datetime, end: datetime) -> TimeSeries:
        """Return the chronologically ordered subset of readings for a sensor
        whose timestamps fall in ``[start, end]``. Unknown sensors return an
        empty series (no exception)."""
        buf = self._readings.get(sensor_id, deque())
        unit = buf[0].unit if buf else ""
        points = [
            TimeSeriesPoint(timestamp=r.timestamp, value=r.value)
            for r in buf
            if start <= r.timestamp <= end
        ]
        return TimeSeries(sensor_id=sensor_id, unit=unit, points=points)

    def latest(self, sensor_ids: list[str]) -> dict[str, SensorReading]:
        """Return the most recent reading per sensor, omitting sensors that
        have never been ingested."""
        result: dict[str, SensorReading] = {}
        for sid in sensor_ids:
            buf = self._readings.get(sid)
            if buf:
                result[sid] = buf[-1]
        return result

    def subsystem_state(
        self, subsystem: str, metadata: SensorMetadata
    ) -> SubsystemSnapshot:
        """Latest reading per sensor in the requested subsystem.

        Subsystem membership is read from ``metadata.sensors[sensor_id]["subsystem"]``.
        The lookup tolerates either the long-form names advertised in the LLM
        tool schema (``thermal_control_system``) or the short codes that EDEN
        ISS actually uses (``TCS``) — both sides are normalised through the
        alias map. Sensors with no data yet are omitted from ``readings`` and
        ``units``.
        """
        target = _normalise_subsystem(subsystem)
        subsystem_sensors = [
            sid
            for sid, info in metadata.sensors.items()
            if _normalise_subsystem(str(info.get("subsystem", ""))) == target
        ]
        latest = self.latest(subsystem_sensors)

        if latest:
            snapshot_ts = max(r.timestamp for r in latest.values())
        else:
            snapshot_ts = datetime.min.replace(tzinfo=timezone.utc)

        return SubsystemSnapshot(
            subsystem=subsystem,
            timestamp=snapshot_ts,
            readings={sid: r.value for sid, r in latest.items()},
            units={sid: r.unit for sid, r in latest.items()},
        )
