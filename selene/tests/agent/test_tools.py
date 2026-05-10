"""Tests for selene.agent.tools.

Each tool is exercised with synthetic store contents and a small in-memory KB.
Determinism is asserted by running each tool twice on freshly-built inputs and
comparing outputs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from selene.agent.store import TelemetryStore
from selene.agent.tools import (
    compute_severity,
    correlate_signals,
    draft_ground_communication,
    draft_workorder,
    fetch_subsystem_state,
    lookup_failure_mode,
    query_sensor_history,
)
from selene.agent.types import Diagnosis, SeverityScore
from selene.core.interfaces import (
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
)
from selene.knowledge.models import Citation, FailureMode, Signature


_T0 = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
_DT = timedelta(minutes=5)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _frame(i: int, values: dict[str, tuple[float, str]]) -> TelemetryFrame:
    ts = _T0 + i * _DT
    readings = {
        sid: SensorReading(sensor_id=sid, timestamp=ts, value=v, unit=u)
        for sid, (v, u) in values.items()
    }
    return TelemetryFrame(timestamp=ts, readings=readings)


def _build_store() -> TelemetryStore:
    """Two anti-correlated sensors plus a thermal sensor."""
    store = TelemetryStore(retention=timedelta(hours=24))
    for i in range(20):
        store.ingest(
            _frame(
                i,
                {
                    "ams-feg/co2-1": (400.0 + i, "ppm"),
                    "ams-feg/rh-1": (60.0 - i, "%"),
                    "tcs/loop-press-1": (1000.0 - 0.1 * i, "kPa"),
                },
            )
        )
    return store


def _metadata() -> SensorMetadata:
    return SensorMetadata(
        sensors={
            "ams-feg/co2-1": {"subsystem": "atmosphere_management_system", "unit": "ppm"},
            "ams-feg/rh-1": {"subsystem": "atmosphere_management_system", "unit": "%"},
            "tcs/loop-press-1": {"subsystem": "thermal_control_system", "unit": "kPa"},
        },
        subsystems=["atmosphere_management_system", "thermal_control_system"],
        sampling_rate_seconds=300.0,
    )


def _kb() -> dict[str, FailureMode]:
    cita = Citation(
        source_type="NTRS", identifier="20190029027", title="ISS P1 EATCS", url=None
    )
    return {
        "iss_p1_eatcs_leak_2011": FailureMode(
            id="iss_p1_eatcs_leak_2011",
            name="ISS P1 EATCS leak",
            affected_subsystems=["thermal_control_system"],
            primary_signature=[
                Signature(
                    sensor_pattern="thermal_loop_pressure",
                    pattern_type="slow_drift",
                    direction="decreasing",
                    time_scale="weeks",
                )
            ],
            secondary_signature=[],
            typical_onset="weeks",
            distinguishing_features=["pressure decay accelerates"],
            differential_diagnosis=["sensor_calibration_drift"],
            historical_context="2011 detection",
            citations=[cita],
            typical_response="cross-feed loops",
        ),
        "co2_drift": FailureMode(
            id="co2_drift",
            name="AMS CO2 drift",
            affected_subsystems=["atmosphere_management_system"],
            primary_signature=[
                Signature(
                    sensor_pattern="atmosphere_co2",
                    pattern_type="slow_drift",
                    direction="increasing",
                    time_scale="hours",
                )
            ],
            secondary_signature=[],
            typical_onset="hours",
            distinguishing_features=["baseline rises"],
            differential_diagnosis=[],
            historical_context="synthetic",
            citations=[cita],
            typical_response="schedule scrubber regen",
        ),
    }


def _diagnosis_thermal() -> Diagnosis:
    return Diagnosis(
        primary_hypothesis="Slow ammonia leak in thermal loop A",
        confidence=0.72,
        matched_failure_modes=["iss_p1_eatcs_leak_2011"],
        supporting_evidence=["loop pressure decay", "valve drift"],
        differential_hypotheses=["sensor calibration drift"],
        recommended_actions=[
            "Reconfigure loops to cross-feed from Loop B",
            "Inspect valve actuator on Loop A",
        ],
        citations=[
            Citation(
                source_type="NTRS",
                identifier="20190029027",
                title="ISS P1 EATCS leak",
                url=None,
            )
        ],
    )


# ── query_sensor_history ─────────────────────────────────────────────────────


class TestQuerySensorHistory:
    def test_returns_chronological_points(self):
        store = _build_store()
        ts = asyncio.run(
            query_sensor_history(
                "ams-feg/co2-1", _T0, _T0 + 5 * _DT, store=store
            )
        )
        assert [p.value for p in ts.points] == [400.0, 401.0, 402.0, 403.0, 404.0, 405.0]
        assert ts.unit == "ppm"

    def test_unknown_sensor_returns_empty_series(self):
        store = _build_store()
        ts = asyncio.run(
            query_sensor_history("nope", _T0, _T0 + 5 * _DT, store=store)
        )
        assert ts.points == []
        assert ts.sensor_id == "nope"

    def test_deterministic(self):
        a = asyncio.run(
            query_sensor_history(
                "ams-feg/co2-1", _T0, _T0 + 5 * _DT, store=_build_store()
            )
        )
        b = asyncio.run(
            query_sensor_history(
                "ams-feg/co2-1", _T0, _T0 + 5 * _DT, store=_build_store()
            )
        )
        assert a == b


# ── fetch_subsystem_state ─────────────────────────────────────────────────────


class TestFetchSubsystemState:
    def test_atmosphere_subsystem_snapshot(self):
        snap = asyncio.run(
            fetch_subsystem_state(
                "atmosphere_management_system",
                store=_build_store(),
                metadata=_metadata(),
            )
        )
        assert set(snap.readings) == {"ams-feg/co2-1", "ams-feg/rh-1"}
        assert snap.timestamp == _T0 + 19 * _DT

    def test_unknown_subsystem_is_empty(self):
        snap = asyncio.run(
            fetch_subsystem_state(
                "made_up_system", store=_build_store(), metadata=_metadata()
            )
        )
        assert snap.readings == {}


# ── correlate_signals ─────────────────────────────────────────────────────────


class TestCorrelateSignals:
    def test_anti_correlated_pair_has_negative_r(self):
        rep = asyncio.run(
            correlate_signals(
                ["ams-feg/co2-1", "ams-feg/rh-1"],
                window=timedelta(hours=2),
                store=_build_store(),
            )
        )
        assert len(rep.correlations) == 1
        c = rep.correlations[0]
        assert c.sensor_a == "ams-feg/co2-1"
        assert c.sensor_b == "ams-feg/rh-1"
        assert c.pearson_r == pytest.approx(-1.0, abs=1e-6)
        assert c.lag_seconds == 0.0

    def test_three_sensors_yield_three_pairs(self):
        rep = asyncio.run(
            correlate_signals(
                ["ams-feg/co2-1", "ams-feg/rh-1", "tcs/loop-press-1"],
                window=timedelta(hours=2),
                store=_build_store(),
            )
        )
        assert len(rep.correlations) == 3
        pairs = {(c.sensor_a, c.sensor_b) for c in rep.correlations}
        assert pairs == {
            ("ams-feg/co2-1", "ams-feg/rh-1"),
            ("ams-feg/co2-1", "tcs/loop-press-1"),
            ("ams-feg/rh-1", "tcs/loop-press-1"),
        }

    def test_single_sensor_returns_empty_correlations(self):
        rep = asyncio.run(
            correlate_signals(
                ["ams-feg/co2-1"], window=timedelta(hours=2), store=_build_store()
            )
        )
        assert rep.correlations == []
        assert rep.window_end > rep.window_start

    def test_empty_store_returns_deterministic_window(self):
        empty_store = TelemetryStore()
        a = asyncio.run(
            correlate_signals(
                ["x", "y"], window=timedelta(hours=1), store=empty_store
            )
        )
        b = asyncio.run(
            correlate_signals(
                ["x", "y"], window=timedelta(hours=1), store=TelemetryStore()
            )
        )
        assert a == b
        assert a.correlations == []

    def test_deterministic_on_real_data(self):
        a = asyncio.run(
            correlate_signals(
                ["ams-feg/co2-1", "ams-feg/rh-1"],
                window=timedelta(hours=2),
                store=_build_store(),
            )
        )
        b = asyncio.run(
            correlate_signals(
                ["ams-feg/co2-1", "ams-feg/rh-1"],
                window=timedelta(hours=2),
                store=_build_store(),
            )
        )
        assert a == b


# ── lookup_failure_mode ──────────────────────────────────────────────────────


class TestLookupFailureMode:
    def test_thermal_symptoms_rank_thermal_first(self):
        symptoms = [
            Signature(
                sensor_pattern="thermal_loop_pressure",
                pattern_type="slow_drift",
                direction="decreasing",
                time_scale="weeks",
            )
        ]
        results = asyncio.run(lookup_failure_mode(symptoms, kb=_kb()))
        assert results[0][0].id == "iss_p1_eatcs_leak_2011"

    def test_atmosphere_symptoms_rank_co2_first(self):
        symptoms = [
            Signature(
                sensor_pattern="atmosphere_co2",
                pattern_type="slow_drift",
                direction="increasing",
                time_scale="hours",
            )
        ]
        results = asyncio.run(lookup_failure_mode(symptoms, kb=_kb()))
        assert results[0][0].id == "co2_drift"

    def test_empty_kb_returns_empty(self):
        symptoms = [
            Signature(
                sensor_pattern="thermal_loop_pressure",
                pattern_type="slow_drift",
                direction="decreasing",
                time_scale="weeks",
            )
        ]
        assert asyncio.run(lookup_failure_mode(symptoms, kb={})) == []


# ── compute_severity ─────────────────────────────────────────────────────────


class TestComputeSeverity:
    def test_thermal_diagnosis_with_high_confidence_is_critical(self):
        d = _diagnosis_thermal()
        score = asyncio.run(compute_severity(d, context={}))
        assert score.level in {"critical", "warning"}
        # 0.72 * 0.85 = 0.612 → critical (≥0.6)
        assert score.score == pytest.approx(0.612, abs=1e-3)
        assert score.level == "critical"

    def test_explicit_subsystem_override(self):
        d = _diagnosis_thermal()
        score = asyncio.run(
            compute_severity(d, context={"subsystem": "illumination_control_system"})
        )
        # 0.72 * 0.35 = 0.252 → info
        assert score.level == "info"

    def test_comms_blackout_raises_severity(self):
        d = _diagnosis_thermal()
        baseline = asyncio.run(compute_severity(d, context={}))
        elevated = asyncio.run(compute_severity(d, context={"comms_blackout": True}))
        assert elevated.score == pytest.approx(baseline.score + 0.10, abs=1e-6)

    def test_redundancy_lowers_severity(self):
        d = _diagnosis_thermal()
        baseline = asyncio.run(compute_severity(d, context={}))
        relieved = asyncio.run(
            compute_severity(d, context={"redundancy_available": True})
        )
        assert relieved.score == pytest.approx(baseline.score - 0.10, abs=1e-6)

    def test_score_clamped_to_unit_interval(self):
        d = _diagnosis_thermal().model_copy(update={"confidence": 1.0})
        score = asyncio.run(
            compute_severity(
                d, context={"comms_blackout": True, "crew_count": 6}
            )
        )
        assert 0.0 <= score.score <= 1.0

    def test_unknown_subsystem_uses_default_weight(self):
        d = _diagnosis_thermal()
        score = asyncio.run(compute_severity(d, context={"subsystem": "novel_sys"}))
        assert score.score == pytest.approx(0.72 * 0.55, abs=1e-6)

    def test_deterministic(self):
        d = _diagnosis_thermal()
        a = asyncio.run(compute_severity(d, context={"comms_blackout": True}))
        b = asyncio.run(compute_severity(d, context={"comms_blackout": True}))
        assert a == b


# ── draft_workorder ──────────────────────────────────────────────────────────


class TestDraftWorkorder:
    def test_inspection_action_yields_inspect_kind(self):
        wo = asyncio.run(
            draft_workorder(_diagnosis_thermal(), "Inspect valve actuator on Loop A")
        )
        assert wo.estimated_duration_minutes == 20
        assert wo.required_tools == []
        assert wo.subsystem == "thermal_control_system"

    def test_replace_action_requires_tools(self):
        wo = asyncio.run(
            draft_workorder(_diagnosis_thermal(), "Replace pump flow control assembly")
        )
        assert wo.estimated_duration_minutes == 60
        assert "spare_part" in wo.required_tools
        assert "torque_wrench" in wo.required_tools

    def test_steps_split_on_sentence_boundaries(self):
        wo = asyncio.run(
            draft_workorder(
                _diagnosis_thermal(),
                "Isolate Loop A. Cross-feed from Loop B; verify flow.",
            )
        )
        assert len(wo.steps) == 3

    def test_single_step_fallback(self):
        wo = asyncio.run(draft_workorder(_diagnosis_thermal(), "patrol"))
        assert wo.steps == ["patrol"]

    def test_subsystem_inferred_from_failure_mode_id(self):
        wo = asyncio.run(
            draft_workorder(_diagnosis_thermal(), "review trends")
        )
        assert wo.subsystem == "thermal_control_system"


# ── draft_ground_communication ───────────────────────────────────────────────


class TestDraftGroundCommunication:
    def test_emergency_severity_maps_to_immediate(self):
        d = _diagnosis_thermal()
        sev = SeverityScore(level="emergency", score=0.95, rationale="...")
        comms = asyncio.run(draft_ground_communication(d, sev))
        assert comms.urgency == "immediate"
        assert "EMERGENCY" in comms.subject

    def test_info_severity_maps_to_routine(self):
        d = _diagnosis_thermal()
        sev = SeverityScore(level="info", score=0.1, rationale="...")
        comms = asyncio.run(draft_ground_communication(d, sev))
        assert comms.urgency == "routine"

    def test_body_includes_evidence_and_actions(self):
        d = _diagnosis_thermal()
        sev = SeverityScore(level="critical", score=0.7, rationale="r")
        comms = asyncio.run(draft_ground_communication(d, sev))
        assert "loop pressure decay" in comms.body
        assert "Reconfigure loops" in comms.body
        assert "iss_p1_eatcs_leak_2011" in comms.body

    def test_body_capped_at_2000_chars(self):
        d = _diagnosis_thermal().model_copy(
            update={"supporting_evidence": ["x" * 3000]}
        )
        sev = SeverityScore(level="critical", score=0.7, rationale="r")
        comms = asyncio.run(draft_ground_communication(d, sev))
        assert len(comms.body) <= 2000
        assert comms.body.endswith("...")

    def test_deterministic(self):
        d = _diagnosis_thermal()
        sev = SeverityScore(level="critical", score=0.7, rationale="r")
        a = asyncio.run(draft_ground_communication(d, sev))
        b = asyncio.run(draft_ground_communication(d, sev))
        assert a == b
