"""Integration test for selene-replay CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures" / "eden_iss_sample"
SCENARIO = Path(__file__).parent.parent.parent / "scenarios" / "test_step_change.yaml"

# The fixture covers 2020-06-01 00:05 – 00:25 UTC (5 frames, 300 s apart).
# The step-change anomaly window in test_step_change.yaml is 02:05–04:05 UTC,
# which is entirely outside the fixture range → all 5 frames should be unmodified.


def _run_replay(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "selene.cli.replay",
            "--data-path", str(FIXTURE),
            "--speed", "max",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


class TestReplayCLI:
    def test_outputs_one_json_line_per_frame(self):
        result = _run_replay()
        lines = [l for l in result.stdout.strip().splitlines() if l]
        assert len(lines) == 5, f"Expected 5 frames, got {len(lines)}"

    def test_each_line_is_valid_telemetry_frame(self):
        result = _run_replay()
        for line in result.stdout.strip().splitlines():
            frame = json.loads(line)
            assert "timestamp" in frame
            assert "readings" in frame

    def test_timestamps_monotonically_increasing(self):
        result = _run_replay()
        timestamps = [
            json.loads(l)["timestamp"]
            for l in result.stdout.strip().splitlines()
        ]
        assert timestamps == sorted(timestamps)

    def test_stderr_reports_sensor_count(self):
        result = _run_replay()
        assert "sensors loaded: 4" in result.stderr

    def test_with_scenario_logs_ground_truth(self):
        result = _run_replay("--scenario", str(SCENARIO))
        assert "test_step_change" in result.stderr
        assert "tcs/temp-ams_in" in result.stderr
        assert "ground truth window" in result.stderr

    def test_with_scenario_still_emits_all_frames(self):
        result = _run_replay("--scenario", str(SCENARIO))
        lines = [l for l in result.stdout.strip().splitlines() if l]
        assert len(lines) == 5

    def test_start_end_filter_frames(self):
        result = _run_replay(
            "--start", "2020-06-01T00:10:00Z",
            "--end",   "2020-06-01T00:20:00Z",
        )
        lines = [l for l in result.stdout.strip().splitlines() if l]
        # 00:10, 00:15, 00:20 → 3 frames
        assert len(lines) == 3

    def test_speed_max_flag(self):
        """'max' speed should not error and should emit frames."""
        result = _run_replay("--speed", "max")
        lines = [l for l in result.stdout.strip().splitlines() if l]
        assert len(lines) == 5

    def test_entrypoint_script(self):
        """selene-replay entry point must be importable and callable."""
        result = subprocess.run(
            ["selene-replay", "--data-path", str(FIXTURE), "--speed", "max"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l]
        assert len(lines) == 5
