"""EDEN ISS telemetry replayer."""

from __future__ import annotations

import asyncio
import glob
import logging
import warnings
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import AsyncIterator

import pandas as pd

from selene.core.interfaces import (
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
)

logger = logging.getLogger(__name__)


class EdenIssReplayer:
    """Replays EDEN ISS 2020 telemetry as a stream of TelemetryFrame objects.

    Args:
        data_path: Path to the dataset root (the directory containing
            ``edeniss2020.csv`` and the per-subsystem sub-directories).
        start_time: First timestamp to include. Defaults to the earliest
            timestamp in the loaded data.
        end_time: Last timestamp to include (inclusive). Defaults to the
            latest timestamp in the loaded data.
        speed_multiplier: Controls replay cadence relative to wall time.
            ``1.0`` = real-time (5-min data interval → 5-min wall time).
            ``60.0`` = 1 minute of data per real second.
            ``None`` = as fast as possible (no sleep between frames).
    """

    def __init__(
        self,
        data_path: Path,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        speed_multiplier: float | None = 1.0,
    ) -> None:
        self._data_path = Path(data_path)
        self._speed_multiplier = speed_multiplier

        self._wide: pd.DataFrame  # wide frame: index=time, columns=sensor_ids
        self._metadata: SensorMetadata
        self._wide, self._metadata = self._load(self._data_path)

        # Apply time filter
        if start_time is not None:
            # Make tz-naive for comparison (timestamps stored as UTC-naive)
            st = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
            self._wide = self._wide[self._wide.index >= st]
        if end_time is not None:
            et = end_time.replace(tzinfo=None) if end_time.tzinfo else end_time
            self._wide = self._wide[self._wide.index <= et]

        if self._wide.empty:
            raise ValueError(
                f"No data in the requested time range "
                f"[{start_time}, {end_time}] for data at {data_path}"
            )

    # ------------------------------------------------------------------
    # TelemetrySource protocol
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[TelemetryFrame]:
        """Yield TelemetryFrame objects in chronological order."""
        prev_ts: datetime | None = None

        for ts, row in self._wide.iterrows():
            frame = self._row_to_frame(ts, row)  # type: ignore[arg-type]
            if frame is None:
                continue

            if self._speed_multiplier is not None and prev_ts is not None:
                # Compute how long to sleep based on data interval and multiplier
                data_interval = (ts - prev_ts).total_seconds()  # type: ignore[operator]
                sleep_secs = data_interval / self._speed_multiplier
                if sleep_secs > 0:
                    await asyncio.sleep(sleep_secs)

            yield frame
            prev_ts = ts  # type: ignore[assignment]

    def get_metadata(self) -> SensorMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_frame(
        self, ts: pd.Timestamp, row: pd.Series
    ) -> TelemetryFrame | None:
        readings: dict[str, SensorReading] = {}
        sensor_info = self._metadata.sensors

        for sensor_id, value in row.items():
            if pd.isna(value):
                logger.warning("NaN value for sensor %s at %s — skipping sensor in frame", sensor_id, ts)
                continue
            info = sensor_info.get(str(sensor_id), {})
            readings[str(sensor_id)] = SensorReading(
                sensor_id=str(sensor_id),
                timestamp=datetime(
                    ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second,
                    tzinfo=timezone.utc,
                ),
                value=float(value),
                unit=info.get("unit", ""),
            )

        if not readings:
            return None

        ts_dt = datetime(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second,
            tzinfo=timezone.utc,
        )
        return TelemetryFrame(timestamp=ts_dt, readings=readings)

    @staticmethod
    def _load(data_path: Path) -> tuple[pd.DataFrame, SensorMetadata]:
        """Load all per-sensor CSVs and the sensor index into memory."""
        index_path = data_path / "edeniss2020.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Sensor index not found: {index_path}")

        index_df = pd.read_csv(index_path)

        # Build sensor metadata dict keyed by canonical sensor_id = Path without .csv
        sensors: dict[str, dict] = {}
        subsystem_set: set[str] = set()

        for _, row in index_df.iterrows():
            raw_path: str = str(row["Path"])
            sensor_id = raw_path.removesuffix(".csv")
            subsystem = str(row["Subsystem"])
            unit = str(row["Unit"]) if pd.notna(row["Unit"]) else ""
            sensor_type_short = str(row["Sensor Type (short)"]) if pd.notna(row.get("Sensor Type (short)", float("nan"))) else ""
            sensors[sensor_id] = {
                "unit": unit,
                "subsystem": subsystem,
                "sensor_type": sensor_type_short,
            }
            subsystem_set.add(subsystem)

        # Load all per-sensor CSVs into a single wide DataFrame
        all_frames: list[pd.DataFrame] = []

        for sensor_id, info in sensors.items():
            csv_path = data_path / f"{sensor_id}.csv"
            if not csv_path.exists():
                logger.warning("CSV not found for sensor %s at %s — skipping", sensor_id, csv_path)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(csv_path, parse_dates=["time"])

            # Rename value column to the canonical sensor_id
            value_col = [c for c in df.columns if c != "time"]
            if not value_col:
                continue
            df = df.rename(columns={value_col[0]: sensor_id})
            df = df.set_index("time")
            all_frames.append(df)

        if not all_frames:
            raise ValueError(f"No sensor CSVs loaded from {data_path}")

        # Join all sensors on the shared timestamp grid
        wide = reduce(lambda a, b: a.join(b, how="outer"), all_frames)
        wide = wide.sort_index()

        metadata = SensorMetadata(
            sensors=sensors,
            subsystems=sorted(subsystem_set),
            sampling_rate_seconds=300.0,
        )
        return wide, metadata
