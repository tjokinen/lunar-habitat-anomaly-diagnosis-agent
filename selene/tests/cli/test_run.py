"""Integration tests for selene-run CLI and run_pipeline().

The pipeline is exercised end-to-end on the small fixture dataset with a
mock agent, so no vLLM endpoint is required.  The test verifies:
  - TelemetryFrame events flow through (one per fixture row)
  - AnomalyEvent events are emitted when the threshold is breached
  - AgentEvent trace events are emitted when the agent fires
  - Event ordering: telemetry before detector events within a frame
  - The CLI exits cleanly and emits valid JSON lines
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from selene.agent.agent import ReasoningAgent
from selene.agent.store import TelemetryStore
from selene.agent.types import (
    AgentRunCompleted,
    AgentRunStarted,
    Diagnosis,
    Event,
)
from selene.core.interfaces import AnomalyEvent, SensorReading, TelemetryFrame
from selene.data.replayer import EdenIssReplayer
from selene.detection.threshold import ThresholdDetector
from selene.knowledge.loader import load_kb
from selene.pipeline import run_pipeline

FIXTURE = Path(__file__).parent.parent / "fixtures" / "eden_iss_sample"
SCENARIO = Path(__file__).parent.parent.parent / "scenarios" / "test_step_change.yaml"
CONFIG = Path(__file__).parent.parent.parent / "config" / "sensor_ranges.yaml"


# ---------------------------------------------------------------------------
# Mock agent that fires one AgentRunCompleted without touching the LLM
# ---------------------------------------------------------------------------

def _mock_agent(store: TelemetryStore, kb, metadata) -> ReasoningAgent:
    """Build a ReasoningAgent whose investigate() is fully mocked."""
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.store = store
    agent.kb = kb
    agent.metadata = metadata
    agent.max_iterations = 1

    async def _fake_investigate(trigger: AnomalyEvent, emit: Callable) -> Diagnosis:
        run_id = "test-run-id"
        await emit(AgentRunStarted(
            run_id=run_id,
            timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
            trigger=trigger,
        ))
        diagnosis = Diagnosis(
            primary_hypothesis="test hypothesis",
            confidence=0.9,
            matched_failure_modes=["iss_p1_eatcs_leak_2011"],
            supporting_evidence=["sensor drift observed"],
            recommended_actions=["inspect thermal loop"],
            citations=[],
        )
        await emit(AgentRunCompleted(
            run_id=run_id,
            timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
            diagnosis=diagnosis,
        ))
        return diagnosis

    agent.investigate = _fake_investigate  # type: ignore[method-assign]
    return agent


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _setup(self, inject_threshold_violation: bool = False):
        """Return (replayer, store, agent, kb, detectors)."""
        replayer = EdenIssReplayer(FIXTURE, speed_multiplier=None)
        meta = replayer.get_metadata()
        kb = load_kb()
        store = TelemetryStore(retention=timedelta(hours=2))
        agent = _mock_agent(store, kb, meta)

        detectors = []
        if inject_threshold_violation and CONFIG.exists():
            # Use an artificially tight range to guarantee a threshold breach
            detectors.append(ThresholdDetector({"tcs/pressure-ams": (2.0, 3.0)}))

        return replayer, store, agent, kb, detectors

    def test_all_frames_emitted(self):
        replayer, store, agent, kb, _ = self._setup()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=[],
            agent=agent,
            event_handler=handler,
        ))

        frames = [e for e in received if isinstance(e, TelemetryFrame)]
        # Fixture has 5 rows
        assert len(frames) == 5

    def test_frame_timestamps_monotonic(self):
        replayer, store, agent, kb, _ = self._setup()
        frames: list[TelemetryFrame] = []

        async def handler(event: Event) -> None:
            if isinstance(event, TelemetryFrame):
                frames.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=[],
            agent=agent,
            event_handler=handler,
        ))

        ts = [f.timestamp for f in frames]
        assert ts == sorted(ts)

    def test_threshold_breach_emits_anomaly_event(self):
        replayer, store, agent, kb, detectors = self._setup(inject_threshold_violation=True)
        anomalies: list[AnomalyEvent] = []

        async def handler(event: Event) -> None:
            if isinstance(event, AnomalyEvent):
                anomalies.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=detectors,
            agent=agent,
            event_handler=handler,
        ))

        # All fixture pressure-ams values are around 1.0 bar — below range [2.0, 3.0]
        assert len(anomalies) > 0
        assert all(e.detector_name == "threshold" for e in anomalies)

    def test_anomaly_triggers_agent_events(self):
        replayer, store, agent, kb, detectors = self._setup(inject_threshold_violation=True)
        agent_events: list = []

        async def handler(event: Event) -> None:
            if isinstance(event, (AgentRunStarted, AgentRunCompleted)):
                agent_events.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=detectors,
            agent=agent,
            event_handler=handler,
        ))

        started = [e for e in agent_events if isinstance(e, AgentRunStarted)]
        completed = [e for e in agent_events if isinstance(e, AgentRunCompleted)]
        assert len(started) >= 1
        assert len(completed) >= 1

    def test_event_ordering_telemetry_before_detector(self):
        """TelemetryFrame must appear before any AnomalyEvent for the same frame."""
        replayer, store, agent, kb, detectors = self._setup(inject_threshold_violation=True)
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=detectors,
            agent=agent,
            event_handler=handler,
        ))

        # Find first AnomalyEvent and verify a TelemetryFrame precedes it
        first_anomaly_idx = next(
            (i for i, e in enumerate(received) if isinstance(e, AnomalyEvent)), None
        )
        if first_anomaly_idx is not None:
            assert first_anomaly_idx > 0
            assert isinstance(received[first_anomaly_idx - 1], TelemetryFrame)

    def test_dedup_suppresses_repeated_anomalies(self):
        """Same (detector, sensor) pair should fire at most once within cooldown."""
        replayer, store, agent, kb, _ = self._setup()
        # Tight range that all 5 fixture frames will breach
        detectors = [ThresholdDetector({"tcs/pressure-ams": (2.0, 3.0)})]
        anomalies: list[AnomalyEvent] = []

        async def handler(event: Event) -> None:
            if isinstance(event, AnomalyEvent):
                anomalies.append(event)

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=detectors,
            agent=agent,
            event_handler=handler,
            dedup_cooldown=timedelta(hours=1),  # large cooldown: only 1 fires
        ))

        tcs_anomalies = [a for a in anomalies if "tcs/pressure-ams" in a.affected_sensors]
        assert len(tcs_anomalies) == 1

    def test_high_score_investigated_before_low_score(self):
        """With workers=1, the queue must serve the highest-score trigger first
        even if a lower-score trigger was enqueued earlier in the same frame."""
        replayer, store, agent, kb, _ = self._setup()

        # Two detectors that fire on the same frame; the second produces a
        # much higher score than the first.
        class _FixedDetector:
            def __init__(self, name: str, score: float, sensor: str) -> None:
                self.name = name
                self._score = score
                self._sensor = sensor
                self._fired = False

            async def evaluate(self, window):
                if self._fired or not window.frames:
                    return []
                self._fired = True
                return [AnomalyEvent(
                    detector_name=self.name,
                    timestamp=window.frames[-1].timestamp,
                    affected_sensors=[self._sensor],
                    score=self._score,
                    details={},
                )]

        order: list[str] = []

        async def _fake_investigate(trigger, emit):
            order.append(trigger.detector_name)
            return None

        agent.investigate = _fake_investigate  # type: ignore[method-assign]

        # Low-score detector listed first (would win FIFO); high-score second.
        detectors = [
            _FixedDetector("low", 0.5, "tcs/pressure-ams"),
            _FixedDetector("high", 5.0, "tcs/pressure-ams"),
        ]

        async def handler(event: Event) -> None:
            return None

        asyncio.run(run_pipeline(
            telemetry_source=replayer,
            detectors=detectors,  # type: ignore[arg-type]
            agent=agent,
            event_handler=handler,
            max_concurrent_investigations=1,  # serial → ordering is observable
        ))

        assert order == ["high", "low"]


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestRunCLI:
    def _run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, "-m", "selene.cli.run",
                "--data-path", str(FIXTURE),
                "--speed", "max",
                *extra,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_emits_json_lines(self):
        result = self._run_cli()
        lines = [l for l in result.stdout.strip().splitlines() if l]
        assert len(lines) >= 5
        for line in lines:
            json.loads(line)  # must parse

    def test_with_scenario_logs_ground_truth(self):
        result = self._run_cli("--scenario", str(SCENARIO))
        assert "ground truth" in result.stderr

    def test_stderr_reports_detectors(self):
        result = self._run_cli()
        assert "detectors" in result.stderr

    def test_start_end_filter(self):
        result = self._run_cli(
            "--start", "2020-06-01T00:10:00Z",
            "--end",   "2020-06-01T00:20:00Z",
        )
        lines = [l for l in result.stdout.strip().splitlines() if l]
        frames = [json.loads(l) for l in lines if "readings" in l]
        assert len(frames) == 3
