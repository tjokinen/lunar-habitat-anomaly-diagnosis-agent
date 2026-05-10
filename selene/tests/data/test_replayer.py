"""Tests for EdenIssReplayer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from selene.data.replayer import EdenIssReplayer

FIXTURE = Path(__file__).parent.parent / "fixtures" / "eden_iss_sample"

# The fixture has 5 rows spanning 2020-06-01 00:05 – 00:25 UTC
T_FIRST = datetime(2020, 6, 1, 0, 5, 0, tzinfo=timezone.utc)
T_LAST  = datetime(2020, 6, 1, 0, 25, 0, tzinfo=timezone.utc)


async def collect_frames(replayer: EdenIssReplayer) -> list:
    return [frame async for frame in replayer.stream()]


class TestEdenIssReplayerLoad:
    def test_metadata_sensor_count(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        meta = r.get_metadata()
        # Fixture has 4 sensors
        assert len(meta.sensors) == 4

    def test_metadata_subsystems(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        meta = r.get_metadata()
        assert set(meta.subsystems) == {"TCS", "NDS"}

    def test_metadata_sampling_rate(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        assert r.get_metadata().sampling_rate_seconds == 300.0

    def test_metadata_sensor_units(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        meta = r.get_metadata()
        assert meta.sensors["tcs/pressure-ams"]["unit"] == "bar"
        assert meta.sensors["tcs/temp-ams_in"]["unit"] == "degrees celsius"


class TestEdenIssReplayerStream:
    def test_full_range_defaults(self):
        """Omitting start/end should yield all 5 frames."""
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        assert len(frames) == 5

    def test_speed_multiplier_none_no_delay(self):
        """speed_multiplier=None must yield all frames without sleeping."""
        import time
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        t0 = time.monotonic()
        frames = asyncio.run(collect_frames(r))
        elapsed = time.monotonic() - t0
        assert len(frames) == 5
        # Should complete well under 1 second
        assert elapsed < 1.0

    def test_timestamps_monotonically_increasing(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        ts = [f.timestamp for f in frames]
        assert ts == sorted(ts)
        assert len(set(ts)) == len(ts)

    def test_first_and_last_timestamps(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        assert frames[0].timestamp == T_FIRST
        assert frames[-1].timestamp == T_LAST

    def test_frame_contains_all_sensors(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        frame = frames[0]
        assert "tcs/pressure-ams" in frame.readings
        assert "tcs/temp-ams_in" in frame.readings
        assert "nds/flow-rate-01" in frame.readings

    def test_sensor_reading_values(self):
        r = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        reading = frames[0].readings["tcs/pressure-ams"]
        assert reading.value == pytest.approx(1.0)
        assert reading.unit == "bar"
        assert reading.sensor_id == "tcs/pressure-ams"


class TestEdenIssReplayerTimeFilter:
    def test_start_time_filters_early_rows(self):
        cutoff = datetime(2020, 6, 1, 0, 15, 0, tzinfo=timezone.utc)
        r = EdenIssReplayer(FIXTURE, start_time=cutoff, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        # rows at 00:15, 00:20, 00:25 → 3 frames
        assert len(frames) == 3
        assert frames[0].timestamp == cutoff

    def test_end_time_filters_late_rows(self):
        cutoff = datetime(2020, 6, 1, 0, 15, 0, tzinfo=timezone.utc)
        r = EdenIssReplayer(FIXTURE, end_time=cutoff, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        # rows at 00:05, 00:10, 00:15 → 3 frames
        assert len(frames) == 3
        assert frames[-1].timestamp == cutoff

    def test_start_and_end_time(self):
        start = datetime(2020, 6, 1, 0, 10, 0, tzinfo=timezone.utc)
        end   = datetime(2020, 6, 1, 0, 20, 0, tzinfo=timezone.utc)
        r = EdenIssReplayer(FIXTURE, start_time=start, end_time=end, speed_multiplier=None)
        frames = asyncio.run(collect_frames(r))
        assert len(frames) == 3

    def test_out_of_range_raises(self):
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="No data"):
            EdenIssReplayer(FIXTURE, start_time=far_future, speed_multiplier=None)
