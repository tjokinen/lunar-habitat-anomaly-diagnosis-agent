"""Tests for ThermalLoopCoolantLeakAnomaly.

Verifies the four properties from the PLAN spec:
- Pre-anomaly frames are unmodified.
- Post-anomaly-end frames are unmodified.
- Mid-anomaly: pressure drops, temperatures rise after lag, valve steps up.
- Ground truth metadata reflects the configured window and sensor list.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from selene.core.interfaces import SensorReading, TelemetryFrame
from selene.scenarios.loader import load_scenario
from selene.scenarios.modules.thermal_leak import (
    ThermalLoopCoolantLeakAnomaly,
)


_T0 = datetime(2020, 6, 1, 2, 0, 0, tzinfo=timezone.utc)
_DURATION_SEC = 28800  # 8h


SCENARIO_YAML = Path(__file__).parent.parent.parent / "scenarios" / "thermal_loop_coolant_leak.yaml"


def _frame(elapsed: timedelta, values: dict[str, float]) -> TelemetryFrame:
    """Build a frame at _T0 + elapsed with the given sensor values (units fixed)."""
    units = {
        "tcs/pressure-ams": "bar",
        "tcs/temp-ams_in": "degrees celsius",
        "tcs/temp-ams_out": "degrees celsius",
        "tcs/valve-ams": "percent",
    }
    ts = _T0 + elapsed
    readings = {
        sid: SensorReading(sensor_id=sid, timestamp=ts, value=v, unit=units[sid])
        for sid, v in values.items()
    }
    return TelemetryFrame(timestamp=ts, readings=readings)


def _nominal_values() -> dict[str, float]:
    return {
        "tcs/pressure-ams": 4.0,           # bar
        "tcs/temp-ams_in": 20.0,           # °C
        "tcs/temp-ams_out": 22.0,          # °C
        "tcs/valve-ams": 30.0,             # %
    }


def _module() -> ThermalLoopCoolantLeakAnomaly:
    return ThermalLoopCoolantLeakAnomaly(
        name="thermal_loop_coolant_leak",
        description="test scenario",
        affected_sensors={
            "pressure": "tcs/pressure-ams",
            "temperatures": ["tcs/temp-ams_in", "tcs/temp-ams_out"],
            "valve": "tcs/valve-ams",
        },
        start_time=_T0,
        duration=_DURATION_SEC,
    )


# ── Time gating ──────────────────────────────────────────────────────────────


class TestTimeGating:
    def test_pre_anomaly_frame_is_unmodified(self):
        mod = _module()
        frame = _frame(timedelta(minutes=-5), _nominal_values())
        out = mod.transform(frame)
        for sid, v in _nominal_values().items():
            assert out.readings[sid].value == v

    def test_post_anomaly_end_frame_is_unmodified(self):
        mod = _module()
        frame = _frame(timedelta(seconds=_DURATION_SEC + 1), _nominal_values())
        out = mod.transform(frame)
        for sid, v in _nominal_values().items():
            assert out.readings[sid].value == v

    def test_applies_at_window_boundaries(self):
        mod = _module()
        assert mod.applies_at(_T0)
        assert mod.applies_at(_T0 + timedelta(seconds=_DURATION_SEC))
        assert not mod.applies_at(_T0 - timedelta(seconds=1))
        assert not mod.applies_at(_T0 + timedelta(seconds=_DURATION_SEC + 1))


# ── Mid-anomaly cascade ──────────────────────────────────────────────────────


class TestMidAnomalyCascade:
    def test_pressure_drops_at_one_hour(self):
        mod = _module()
        frame = _frame(timedelta(hours=1), _nominal_values())
        out = mod.transform(frame)
        assert out.readings["tcs/pressure-ams"].value < 4.0
        # Default initial rate 0.005/h means ≈0.5% loss → about 3.97 bar
        assert out.readings["tcs/pressure-ams"].value > 3.95

    def test_pressure_drop_accelerates(self):
        """Loss curve must be super-linear: drop at 8h ≫ 8 × drop at 1h."""
        mod = _module()
        nominal = _nominal_values()

        out_1h = mod.transform(_frame(timedelta(hours=1), nominal))
        out_8h = mod.transform(
            _frame(timedelta(seconds=_DURATION_SEC), nominal)
        )
        loss_1h = 4.0 - out_1h.readings["tcs/pressure-ams"].value
        loss_8h = 4.0 - out_8h.readings["tcs/pressure-ams"].value

        # Linear-only would give loss_8h = 8 * loss_1h. With acceleration_factor
        # = 1.5, the end-point loss is at least 1.5× the linear extrapolation.
        assert loss_8h > 8.0 * loss_1h

    def test_temperature_unchanged_inside_lag_window(self):
        """Within the compensation lag (30 min by default), temps don't move."""
        mod = _module()
        frame = _frame(timedelta(minutes=15), _nominal_values())
        out = mod.transform(frame)
        assert out.readings["tcs/temp-ams_in"].value == 20.0
        assert out.readings["tcs/temp-ams_out"].value == 22.0

    def test_temperature_rises_after_lag(self):
        mod = _module()
        frame = _frame(timedelta(hours=4), _nominal_values())
        out = mod.transform(frame)
        assert out.readings["tcs/temp-ams_in"].value > 20.0
        assert out.readings["tcs/temp-ams_out"].value > 22.0

    def test_temperature_rise_proportional_to_loss(self):
        """At t=8h the drift should be ≈ k × loss_fraction. With defaults that's
        ~3°C — not half the diurnal range, but well above noise."""
        mod = _module()
        nominal = _nominal_values()
        out = mod.transform(_frame(timedelta(seconds=_DURATION_SEC), nominal))
        loss_fraction = (4.0 - out.readings["tcs/pressure-ams"].value) / 4.0
        expected_rise = 50.0 * loss_fraction
        actual_rise = out.readings["tcs/temp-ams_in"].value - 20.0
        assert actual_rise == pytest.approx(expected_rise, rel=1e-9)

    def test_valve_steps_up_with_loss(self):
        """Valve opening should be monotonically non-decreasing as elapsed time
        grows, with at least one step taken by the end of the window."""
        mod = _module()
        nominal = _nominal_values()
        valve_at = lambda h: mod.transform(  # noqa: E731
            _frame(timedelta(hours=h), nominal)
        ).readings["tcs/valve-ams"].value

        v_30min = valve_at(0.5)
        v_4h = valve_at(4.0)
        v_8h = valve_at(_DURATION_SEC / 3600.0)

        assert v_30min == 30.0  # below 1% loss threshold
        assert v_4h > v_30min   # at least one step engaged
        assert v_8h >= v_4h     # monotone

    def test_valve_clamped_at_100(self):
        """Even with extreme accumulated steps, the valve never exceeds 100%."""
        mod = ThermalLoopCoolantLeakAnomaly(
            name="x",
            description="x",
            affected_sensors={
                "pressure": "tcs/pressure-ams",
                "temperatures": ["tcs/temp-ams_in"],
                "valve": "tcs/valve-ams",
            },
            start_time=_T0,
            duration=_DURATION_SEC,
            valve_step_thresholds=[0.001, 0.002, 0.003, 0.004],
            valve_step_increments=[40.0, 40.0, 40.0, 40.0],
        )
        frame = _frame(timedelta(seconds=_DURATION_SEC), _nominal_values())
        out = mod.transform(frame)
        assert out.readings["tcs/valve-ams"].value == 100.0


# ── Ground truth ─────────────────────────────────────────────────────────────


class TestGroundTruth:
    def test_ground_truth_metadata_correct(self):
        mod = _module()
        gt = mod.get_ground_truth()
        assert gt.scenario_id == "thermal_loop_coolant_leak"
        assert gt.start_time == _T0
        assert gt.end_time == _T0 + timedelta(seconds=_DURATION_SEC)
        assert set(gt.affected_sensors) == {
            "tcs/pressure-ams",
            "tcs/temp-ams_in",
            "tcs/temp-ams_out",
            "tcs/valve-ams",
        }
        assert gt.description == "test scenario"

    def test_affected_sensors_is_flat_list(self):
        """``AnomalyModule.affected_sensors`` must be a flat list of IDs (the
        protocol shape), not the nested config object."""
        mod = _module()
        assert isinstance(mod.affected_sensors, list)
        assert all(isinstance(s, str) for s in mod.affected_sensors)


# ── Frame-passthrough hygiene ────────────────────────────────────────────────


class TestFramePassthrough:
    def test_unrelated_sensors_pass_through_unmodified(self):
        """Sensors not mentioned in ``affected_sensors`` are left alone."""
        mod = _module()
        ts = _T0 + timedelta(hours=4)
        readings = {
            **{
                sid: SensorReading(
                    sensor_id=sid, timestamp=ts, value=v, unit="x"
                )
                for sid, v in _nominal_values().items()
            },
            "ams-feg/co2-1": SensorReading(
                sensor_id="ams-feg/co2-1", timestamp=ts, value=410.0, unit="ppm"
            ),
        }
        frame = TelemetryFrame(timestamp=ts, readings=readings)
        out = mod.transform(frame)
        assert out.readings["ams-feg/co2-1"].value == 410.0

    def test_missing_pressure_sensor_is_silently_ignored(self):
        """If the pressure sensor is missing from a frame, transform should not
        crash — the data set sometimes lacks rows."""
        mod = _module()
        ts = _T0 + timedelta(hours=4)
        readings = {
            "tcs/temp-ams_in": SensorReading(
                sensor_id="tcs/temp-ams_in",
                timestamp=ts,
                value=20.0,
                unit="degrees celsius",
            )
        }
        frame = TelemetryFrame(timestamp=ts, readings=readings)
        out = mod.transform(frame)
        assert "tcs/pressure-ams" not in out.readings
        # Temperature still drifts (lag has elapsed)
        assert out.readings["tcs/temp-ams_in"].value > 20.0


# ── YAML loader integration ──────────────────────────────────────────────────


class TestYAMLLoader:
    def test_loads_from_repo_yaml(self):
        mod = load_scenario(SCENARIO_YAML)
        assert isinstance(mod, ThermalLoopCoolantLeakAnomaly)
        assert mod.name == "thermal_loop_coolant_leak"

    def test_yaml_affected_sensors_match_eden_iss_tcs_ams_branch(self):
        mod = load_scenario(SCENARIO_YAML)
        assert "tcs/pressure-ams" in mod.affected_sensors
        assert "tcs/temp-ams_in" in mod.affected_sensors
        assert "tcs/temp-ams_out" in mod.affected_sensors
        assert "tcs/valve-ams" in mod.affected_sensors

    def test_yaml_ground_truth(self):
        mod = load_scenario(SCENARIO_YAML)
        gt = mod.get_ground_truth()
        assert gt.scenario_id == "thermal_loop_coolant_leak"
        assert (gt.end_time - gt.start_time).total_seconds() == 28800
