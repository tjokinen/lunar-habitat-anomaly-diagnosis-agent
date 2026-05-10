"""ThermalLoopCoolantLeakAnomaly — the headline scenario.

Reproduces the *signature pattern* of the ISS P1 EATCS ammonia leak
(NTRS 20190029027, NTRS 20220003097) on EDEN ISS Thermal Control System
sensors:

1. Slow pressure decay on the loop pressure sensor — sub-threshold initially,
   accelerating exponentially over the simulated window.
2. Compensating temperature drift on coupled inlet/outlet temperature sensors,
   starting after a controller compensation lag (default 30 min).
3. Stepwise increase in the loop control valve opening as the controller
   responds to falling pressure.

We are not replicating ISS-scale physics — the magnitudes are tuned for an
EDEN ISS greenhouse-scale loop. We are replicating the *shape* of the cascade
so that the agent's KB-matching path selects the ISS P1 EATCS leak entry.

**Departure from the original PLAN spec.** The PLAN listed a ``power`` field
in ``affected_sensors`` because ISS-style EATCS modeling expected a power-
consumption proxy as backup cooling kicks in. EDEN ISS TCS has *no power
sensors* (see ``docs/eden_iss_format.md``); the closest control-loop response
in the data is the valve opening percentage, which is precisely what the
matching KB entry's secondary signature expects (``thermal_loop_valve_position``,
increasing). The field is therefore named ``valve``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field, model_validator

from selene.core.interfaces import AnomalyGroundTruth, TelemetryFrame
from selene.scenarios.registry import register_module


class _AffectedSensors(BaseModel):
    pressure: str
    temperatures: list[str] = Field(min_length=1)
    valve: str


class ThermalLoopCoolantLeakConfig(BaseModel):
    """YAML-validated config. Sensor IDs are project-canonical (e.g.
    ``tcs/pressure-ams``). All time fields use ISO-8601; magnitude knobs are
    fractional (pressure) or absolute (temperature, valve %).
    """

    name: str
    description: str
    affected_sensors: _AffectedSensors
    start_time: datetime
    duration: int = Field(gt=0, description="Anomaly duration in seconds.")
    initial_pressure_drop_rate: float = Field(
        default=0.005,
        gt=0.0,
        description=(
            "Fractional pressure loss in the first hour of the leak (~0.5% by "
            "default — sub-threshold for naive thresholding)."
        ),
    )
    acceleration_factor: float = Field(
        default=1.5,
        gt=1.0,
        description=(
            "Multiplier on the cumulative loss curve at the end of the window. "
            "1.0 ≈ purely linear; >1 makes the late stage exponentially worse."
        ),
    )
    temperature_lag_seconds: int = Field(
        default=1800,
        ge=0,
        description="Controller compensation lag before temperatures begin to drift.",
    )
    temperature_coupling_k: float = Field(
        default=50.0,
        ge=0.0,
        description=(
            "°C of upward drift per unit fractional pressure loss. Default 50 "
            "means a 4% pressure loss adds ~2°C — within the EDEN ISS TCS "
            "diurnal range so it stays plausibly hidden in noise."
        ),
    )
    valve_step_thresholds: list[float] = Field(
        default=[0.01, 0.02, 0.04, 0.06],
        description="Cumulative pressure-loss fractions at which valve opens further.",
    )
    valve_step_increments: list[float] = Field(
        default=[5.0, 10.0, 20.0, 30.0],
        description="Cumulative percentage points added to the valve at each threshold.",
    )

    @model_validator(mode="after")
    def _validate_step_table(self) -> "ThermalLoopCoolantLeakConfig":
        if len(self.valve_step_thresholds) != len(self.valve_step_increments):
            raise ValueError(
                "valve_step_thresholds and valve_step_increments must have the same length"
            )
        if any(
            b <= a
            for a, b in zip(self.valve_step_thresholds, self.valve_step_thresholds[1:])
        ):
            raise ValueError("valve_step_thresholds must be strictly increasing")
        return self


@register_module("thermal_loop_coolant_leak")
class ThermalLoopCoolantLeakAnomaly:
    """``AnomalyModule`` implementing the slow-leak signature on a TCS branch.

    Sensor IDs are flat in ``self.affected_sensors`` for compatibility with
    the protocol; per-role IDs (pressure / temperatures / valve) are exposed
    as private fields used by ``transform``.
    """

    def __init__(self, **kwargs: object) -> None:
        cfg = ThermalLoopCoolantLeakConfig(**kwargs)  # type: ignore[arg-type]
        self.name: str = cfg.name
        self.description: str = cfg.description

        self._pressure_id = cfg.affected_sensors.pressure
        self._temperature_ids = list(cfg.affected_sensors.temperatures)
        self._valve_id = cfg.affected_sensors.valve
        self.affected_sensors: list[str] = [
            self._pressure_id,
            *self._temperature_ids,
            self._valve_id,
        ]

        self._start = cfg.start_time
        self._duration = cfg.duration
        self._end = cfg.start_time + timedelta(seconds=cfg.duration)
        self._initial_rate = cfg.initial_pressure_drop_rate
        self._accel = cfg.acceleration_factor
        self._temp_lag = cfg.temperature_lag_seconds
        self._temp_k = cfg.temperature_coupling_k
        self._valve_thresholds = list(cfg.valve_step_thresholds)
        self._valve_increments = list(cfg.valve_step_increments)

    # ------------------------------------------------------------------
    # AnomalyModule protocol
    # ------------------------------------------------------------------

    def applies_at(self, t: datetime) -> bool:
        return self._normalize(self._start) <= self._normalize(t) <= self._normalize(self._end)

    def transform(self, frame: TelemetryFrame) -> TelemetryFrame:
        if not self.applies_at(frame.timestamp):
            return frame

        elapsed_seconds = (
            self._normalize(frame.timestamp) - self._normalize(self._start)
        ).total_seconds()
        loss = self._pressure_loss_fraction(elapsed_seconds)

        new_readings = dict(frame.readings)

        # 1) Pressure: multiplicative decay against the underlying value so the
        #    anomaly rides on top of any natural variation already in the data.
        if self._pressure_id in new_readings:
            r = new_readings[self._pressure_id]
            new_readings[self._pressure_id] = r.model_copy(
                update={"value": r.value * (1.0 - loss)}
            )

        # 2) Temperature: additive rise after the compensation lag, magnitude
        #    proportional to cumulative pressure loss.
        if elapsed_seconds >= self._temp_lag:
            for tid in self._temperature_ids:
                if tid in new_readings:
                    r = new_readings[tid]
                    new_readings[tid] = r.model_copy(
                        update={"value": r.value + self._temp_k * loss}
                    )

        # 3) Valve: stepwise increase clamped to the physical 0–100% range.
        valve_increment = self._valve_increment_for(loss)
        if valve_increment > 0.0 and self._valve_id in new_readings:
            r = new_readings[self._valve_id]
            new_readings[self._valve_id] = r.model_copy(
                update={"value": min(100.0, r.value + valve_increment)}
            )

        return frame.model_copy(update={"readings": new_readings})

    def get_ground_truth(self) -> AnomalyGroundTruth:
        return AnomalyGroundTruth(
            scenario_id=self.name,
            start_time=self._start,
            end_time=self._end,
            affected_sensors=self.affected_sensors,
            description=self.description,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pressure_loss_fraction(self, elapsed_seconds: float) -> float:
        """Cumulative fractional pressure loss at ``elapsed_seconds`` into the
        anomaly. Linear at first, accelerating exponentially toward the end of
        the window. Returns 0 outside the active window.

        Shape: ``f(t) = rate * h * accel ^ (h / total_h)`` where ``h = t/3600``
        and ``total_h = duration/3600``. At t=0, f=0. At t=duration, f reaches
        ``rate * total_h * accel`` — the worst-case scenario. With the defaults
        (rate=0.005/h, accel=1.5, duration=8h) this is 6% pressure loss.
        """
        if elapsed_seconds <= 0.0:
            return 0.0
        if elapsed_seconds >= self._duration:
            elapsed_seconds = float(self._duration)
        hours = elapsed_seconds / 3600.0
        total_hours = self._duration / 3600.0
        return self._initial_rate * hours * (self._accel ** (hours / total_hours))

    def _valve_increment_for(self, loss: float) -> float:
        """Total accumulated valve-opening increment for a given fractional
        pressure loss. Sums every step whose threshold is met (so the largest
        threshold reached implies all smaller ones have already contributed)."""
        total = 0.0
        for threshold, inc in zip(self._valve_thresholds, self._valve_increments):
            if loss >= threshold:
                total += inc
            else:
                break
        return total

    @staticmethod
    def _normalize(t: datetime) -> datetime:
        """Strip tz so naive/aware comparisons don't raise. The replayer treats
        EDEN ISS timestamps as UTC; mixing aware/naive across module boundaries
        is otherwise easy to trip on."""
        return t.replace(tzinfo=None) if t.tzinfo else t
