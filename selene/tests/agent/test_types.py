"""Tests for selene.agent.types — schema and discriminated union dispatch."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from selene.agent.types import (
    AgentEvent,
    AgentRunCompleted,
    AgentRunFailed,
    AgentRunStarted,
    CorrelationReport,
    Diagnosis,
    GroundComms,
    HypothesisLadderUpdated,
    SensorCorrelation,
    SeverityScore,
    SubsystemSnapshot,
    TimeSeries,
    TimeSeriesPoint,
    ToolCallCompleted,
    ToolCallStarted,
    WorkOrder,
)
from selene.core.interfaces import AnomalyEvent
from selene.knowledge.models import Citation


_T = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)


def _diagnosis() -> Diagnosis:
    return Diagnosis(
        primary_hypothesis="thermal loop slow leak",
        confidence=0.72,
        matched_failure_modes=["iss_p1_eatcs_leak_2011"],
        supporting_evidence=["pressure decay", "valve drift"],
        differential_hypotheses=["sensor drift"],
        recommended_actions=["isolate loop"],
        citations=[
            Citation(
                source_type="NTRS",
                identifier="20190029027",
                title="ISS P1 EATCS leak (2019)",
                url="https://ntrs.nasa.gov/citations/20190029027",
            )
        ],
    )


# ── Tool I/O models ───────────────────────────────────────────────────────────


class TestToolIO:
    def test_time_series_point_round_trip(self):
        p = TimeSeriesPoint(timestamp=_T, value=1.5)
        assert p.value == 1.5
        assert p.timestamp == _T

    def test_time_series_holds_chronological_points(self):
        ts = TimeSeries(
            sensor_id="ams-feg/co2-1",
            unit="ppm",
            points=[TimeSeriesPoint(timestamp=_T, value=410.0)],
        )
        assert ts.sensor_id == "ams-feg/co2-1"
        assert ts.points[0].value == 410.0

    def test_subsystem_snapshot_keys_match(self):
        snap = SubsystemSnapshot(
            subsystem="atmosphere_management_system",
            timestamp=_T,
            readings={"a": 1.0, "b": 2.0},
            units={"a": "ppm", "b": "%"},
        )
        assert set(snap.readings) == set(snap.units)

    def test_pearson_r_must_be_in_range(self):
        with pytest.raises(ValidationError):
            SensorCorrelation(sensor_a="a", sensor_b="b", pearson_r=1.5)
        with pytest.raises(ValidationError):
            SensorCorrelation(sensor_a="a", sensor_b="b", pearson_r=-1.5)

    def test_correlation_report_minimal(self):
        rep = CorrelationReport(
            sensor_ids=["a", "b"],
            window_start=_T,
            window_end=_T,
            correlations=[],
        )
        assert rep.correlations == []

    def test_severity_score_level_must_be_literal(self):
        with pytest.raises(ValidationError):
            SeverityScore(level="bogus", score=0.5, rationale="...")  # type: ignore[arg-type]

    def test_severity_score_score_in_range(self):
        with pytest.raises(ValidationError):
            SeverityScore(level="info", score=1.1, rationale="...")

    def test_workorder_duration_non_negative(self):
        with pytest.raises(ValidationError):
            WorkOrder(
                title="t", subsystem="s", steps=["step"], estimated_duration_minutes=-1
            )

    def test_ground_comms_body_max_length(self):
        with pytest.raises(ValidationError):
            GroundComms(subject="s", body="x" * 2001, urgency="routine")

    def test_ground_comms_urgency_literal(self):
        with pytest.raises(ValidationError):
            GroundComms(subject="s", body="b", urgency="meh")  # type: ignore[arg-type]


# ── Diagnosis ────────────────────────────────────────────────────────────────


class TestDiagnosis:
    def test_diagnosis_minimum_fields(self):
        d = _diagnosis()
        assert d.confidence == 0.72
        assert d.matched_failure_modes == ["iss_p1_eatcs_leak_2011"]

    def test_confidence_in_range(self):
        with pytest.raises(ValidationError):
            Diagnosis(
                primary_hypothesis="x",
                confidence=1.5,
                matched_failure_modes=[],
                supporting_evidence=[],
                recommended_actions=[],
                citations=[],
            )


# ── Agent trace events / discriminated union ─────────────────────────────────


class TestAgentEventDiscriminator:
    def test_started_event_round_trip(self):
        anomaly = AnomalyEvent(
            detector_name="mdi",
            timestamp=_T,
            affected_sensors=["a"],
            score=3.1,
        )
        ev = AgentRunStarted(run_id="r1", timestamp=_T, trigger=anomaly)
        assert ev.type == "agent_run_started"

    def test_tool_call_started_and_completed_distinguished_by_type(self):
        started = ToolCallStarted(
            run_id="r1",
            timestamp=_T,
            call_id="c1",
            tool_name="query_sensor_history",
            arguments={"sensor_id": "x"},
        )
        completed = ToolCallCompleted(
            run_id="r1",
            timestamp=_T,
            call_id="c1",
            tool_name="query_sensor_history",
            result_summary="empty",
            result={"points": []},
        )
        assert started.type == "tool_call_started"
        assert completed.type == "tool_call_completed"
        assert completed.error is None

    def test_hypothesis_ladder_updated(self):
        ev = HypothesisLadderUpdated(
            run_id="r1", timestamp=_T, ranked=[("iss_p1_eatcs_leak_2011", 0.81)]
        )
        assert ev.ranked[0][1] == 0.81

    def test_run_failed_carries_reason(self):
        ev = AgentRunFailed(run_id="r1", timestamp=_T, reason="max_iterations")
        assert ev.reason == "max_iterations"

    def test_discriminated_union_dispatches_on_type(self):
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        payload = {
            "type": "agent_run_completed",
            "run_id": "r1",
            "timestamp": _T.isoformat(),
            "diagnosis": _diagnosis().model_dump(mode="json"),
        }
        ev = adapter.validate_python(payload)
        assert isinstance(ev, AgentRunCompleted)

    def test_discriminator_rejects_unknown_type(self):
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {"type": "not_a_thing", "run_id": "r1", "timestamp": _T.isoformat()}
            )

    def test_discriminator_round_trip_via_json(self):
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        original = ToolCallStarted(
            run_id="r1",
            timestamp=_T,
            call_id="c1",
            tool_name="x",
            arguments={"k": 1},
        )
        dumped = adapter.dump_python(original, mode="json")
        restored = adapter.validate_python(dumped)
        assert isinstance(restored, ToolCallStarted)
        assert restored.tool_name == "x"
