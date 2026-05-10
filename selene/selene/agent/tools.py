"""Agent-callable tools.

Each tool is a plain async function with typed inputs and a typed return. The
``store``, ``kb``, and ``metadata`` keyword-only parameters are bound at agent
construction time (step 2.7) — the JSON schema the LLM sees through OpenAI
function-calling exposes only the LLM-facing arguments (sensor_id, start, end,
subsystem, sensor_ids, window, symptoms, diagnosis, context, recommended_action,
severity).

Tools are deterministic given the same inputs: no clock reads, no randomness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

from selene.agent.types import (
    CorrelationReport,
    Diagnosis,
    GroundComms,
    SensorCorrelation,
    SeverityScore,
    SubsystemSnapshot,
    TimeSeries,
    WorkOrder,
)
from selene.agent.store import TelemetryStore
from selene.core.interfaces import SensorMetadata
from selene.knowledge.matcher import match_failure_modes
from selene.knowledge.models import FailureMode, Signature

# Deterministic timestamp used when no telemetry exists for the requested
# sensors. Unix epoch is far enough in the past that ``end_ts - window``
# never overflows for any realistic window, and far enough from real EDEN ISS
# data (2020+) that callers can recognize the "no data" case.
_EMPTY_SENTINEL = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Read-side tools — touch the store / KB / metadata
# ---------------------------------------------------------------------------


async def query_sensor_history(
    sensor_id: str,
    start: datetime,
    end: datetime,
    *,
    store: TelemetryStore,
) -> TimeSeries:
    """Return the chronologically ordered readings for ``sensor_id`` between
    ``start`` and ``end`` (inclusive). Unknown sensors return an empty series."""
    return store.history(sensor_id, start, end)


async def fetch_subsystem_state(
    subsystem: str,
    *,
    store: TelemetryStore,
    metadata: SensorMetadata,
) -> SubsystemSnapshot:
    """Latest reading per sensor in ``subsystem``, anchored at the most recent
    timestamp across those sensors."""
    return store.subsystem_state(subsystem, metadata)


async def correlate_signals(
    sensor_ids: list[str],
    window: timedelta,
    *,
    store: TelemetryStore,
) -> CorrelationReport:
    """Pairwise zero-lag Pearson correlation across ``sensor_ids`` over the
    last ``window`` of telemetry.

    Pairs are emitted in the order returned by enumerating ``sensor_ids``
    (i.e. (a, b) with a < b in the input ordering, no self-pairs). Sensors
    with fewer than two overlapping samples in the window are skipped from
    every pair they would have appeared in. ``lag_seconds`` is reported as
    ``0.0``: lag inference is intentionally out of scope here — call multiple
    times with shifted windows if the agent needs lag attribution.
    """
    if len(sensor_ids) < 2:
        # Empty correlations, but we still need a sensible window range.
        latest = store.latest(sensor_ids)
        if latest:
            end_ts = max(r.timestamp for r in latest.values())
        else:
            end_ts = _EMPTY_SENTINEL
        return CorrelationReport(
            sensor_ids=list(sensor_ids),
            window_start=end_ts - window,
            window_end=end_ts,
            correlations=[],
        )

    latest = store.latest(sensor_ids)
    if not latest:
        # Nothing in the store for any requested sensor — degenerate but valid.
        end_ts = _EMPTY_SENTINEL
        return CorrelationReport(
            sensor_ids=list(sensor_ids),
            window_start=end_ts - window,
            window_end=end_ts,
            correlations=[],
        )

    end_ts = max(r.timestamp for r in latest.values())
    start_ts = end_ts - window

    # Build a {timestamp -> {sensor_id -> value}} alignment table over the window.
    aligned: dict[datetime, dict[str, float]] = {}
    for sid in sensor_ids:
        ts = store.history(sid, start_ts, end_ts)
        for p in ts.points:
            aligned.setdefault(p.timestamp, {})[sid] = p.value

    correlations: list[SensorCorrelation] = []
    for i, a in enumerate(sensor_ids):
        for b in sensor_ids[i + 1 :]:
            xs: list[float] = []
            ys: list[float] = []
            for _, row in sorted(aligned.items()):
                if a in row and b in row:
                    xs.append(row[a])
                    ys.append(row[b])
            if len(xs) < 2:
                continue
            r = _safe_pearson(np.asarray(xs), np.asarray(ys))
            correlations.append(
                SensorCorrelation(sensor_a=a, sensor_b=b, pearson_r=r, lag_seconds=0.0)
            )

    return CorrelationReport(
        sensor_ids=list(sensor_ids),
        window_start=start_ts,
        window_end=end_ts,
        correlations=correlations,
    )


async def lookup_failure_mode(
    symptoms: list[Signature],
    *,
    kb: dict[str, FailureMode],
) -> list[tuple[FailureMode, float]]:
    """Rank KB entries by symptom overlap. Thin async wrapper over
    ``selene.knowledge.matcher.match_failure_modes``."""
    return match_failure_modes(symptoms, kb)


# ---------------------------------------------------------------------------
# Synthesis tools — pure functions of their inputs, no I/O
# ---------------------------------------------------------------------------


# Subsystem severity weights. Higher = same confidence translates to a
# higher severity. Thermal and atmosphere are crew-survival adjacent on a
# real life-support stack; nutrient and illumination are crop-survival
# adjacent — slower-onset, more recoverable.
_SUBSYSTEM_WEIGHT: dict[str, float] = {
    "thermal_control_system": 0.85,
    "atmosphere_management_system": 0.80,
    "nutrient_delivery_system": 0.45,
    "illumination_control_system": 0.35,
}
_DEFAULT_SUBSYSTEM_WEIGHT = 0.55

# Substring → subsystem inference, applied to matched failure-mode IDs and
# the primary hypothesis text in that order. First match wins.
_SUBSYSTEM_INFERENCE_RULES: list[tuple[str, str]] = [
    ("thermal", "thermal_control_system"),
    ("eatcs", "thermal_control_system"),
    ("tcs", "thermal_control_system"),
    ("scrubber", "atmosphere_management_system"),
    ("cdra", "atmosphere_management_system"),
    ("co2", "atmosphere_management_system"),
    ("atmosphere", "atmosphere_management_system"),
    ("ams", "atmosphere_management_system"),
    ("pump", "nutrient_delivery_system"),
    ("nutrient", "nutrient_delivery_system"),
    ("nds", "nutrient_delivery_system"),
    ("illumination", "illumination_control_system"),
    ("ics", "illumination_control_system"),
    ("led", "illumination_control_system"),
    ("par", "illumination_control_system"),
]


async def compute_severity(diagnosis: Diagnosis, context: dict) -> SeverityScore:
    """Map a Diagnosis + operational context to a severity level.

    Score is computed deterministically as
    ``confidence * subsystem_weight + context_modifier``, clamped to [0, 1].

    Recognized context keys (all optional):
      - ``subsystem`` (str): override subsystem inference.
      - ``comms_blackout`` (bool): no ground-loop available → +0.10.
      - ``redundancy_available`` (bool): hot spare available → −0.10.
      - ``crew_count`` (int): >4 crew adds +0.05; ≤2 crew adds +0.05 (single
        operator load).
    Unknown keys are ignored.
    """
    subsystem = _infer_subsystem(diagnosis, context)
    weight = _SUBSYSTEM_WEIGHT.get(subsystem, _DEFAULT_SUBSYSTEM_WEIGHT)

    base = diagnosis.confidence * weight

    modifier = 0.0
    modifier_notes: list[str] = []
    if context.get("comms_blackout"):
        modifier += 0.10
        modifier_notes.append("comms blackout (+0.10)")
    if context.get("redundancy_available"):
        modifier -= 0.10
        modifier_notes.append("redundancy available (−0.10)")
    crew_count = context.get("crew_count")
    if isinstance(crew_count, int):
        if crew_count > 4:
            modifier += 0.05
            modifier_notes.append("expanded crew (+0.05)")
        elif crew_count <= 2:
            modifier += 0.05
            modifier_notes.append("minimal crew (+0.05)")

    score = max(0.0, min(1.0, base + modifier))
    level = _score_to_level(score)

    rationale = (
        f"subsystem={subsystem} (weight {weight:.2f}) × "
        f"confidence {diagnosis.confidence:.2f} → base {base:.2f}; "
        f"modifiers: {', '.join(modifier_notes) if modifier_notes else 'none'} "
        f"→ score {score:.2f} → {level}"
    )
    return SeverityScore(level=level, score=score, rationale=rationale)


async def draft_workorder(
    diagnosis: Diagnosis, recommended_action: str
) -> WorkOrder:
    """Render a single recommended action as a structured work order.

    Subsystem and required-tool inference is heuristic — the underlying
    Diagnosis schema does not carry a subsystem field, so the action text and
    matched_failure_modes IDs are scanned for keywords. ``required_tools``
    expands by the same keyword scan.
    """
    subsystem = _infer_subsystem(diagnosis, context={"recommended_action": recommended_action})

    duration, tools = _action_kind(recommended_action)
    steps = _split_action_into_steps(recommended_action)

    title = f"{diagnosis.primary_hypothesis} — {recommended_action[:60]}".rstrip()

    return WorkOrder(
        title=title,
        subsystem=subsystem,
        steps=steps,
        estimated_duration_minutes=duration,
        required_tools=tools,
    )


async def draft_ground_communication(
    diagnosis: Diagnosis, severity: SeverityScore
) -> GroundComms:
    """Render a Diagnosis + SeverityScore as a plain-text ground comms message.

    Body is hard-capped at 2000 chars by the GroundComms schema; this function
    truncates with an ellipsis if the rendered body would exceed that.
    """
    urgency = _severity_to_urgency(severity.level)
    subject = f"[{severity.level.upper()}] {diagnosis.primary_hypothesis}".strip()

    citation_lines = [
        f"  - {c.source_type} {c.identifier}: {c.title}" for c in diagnosis.citations
    ]
    differentials = (
        "\n  - " + "\n  - ".join(diagnosis.differential_hypotheses)
        if diagnosis.differential_hypotheses
        else " (none flagged)"
    )
    actions = "\n  - " + "\n  - ".join(diagnosis.recommended_actions)
    evidence = "\n  - " + "\n  - ".join(diagnosis.supporting_evidence)

    body = (
        f"Severity: {severity.level} (score {severity.score:.2f})\n"
        f"Confidence: {diagnosis.confidence:.2f}\n\n"
        f"Primary hypothesis: {diagnosis.primary_hypothesis}\n"
        f"Matched failure modes: {', '.join(diagnosis.matched_failure_modes) or '(none)'}\n\n"
        f"Supporting evidence:{evidence}\n\n"
        f"Differential hypotheses:{differentials}\n\n"
        f"Recommended actions:{actions}\n\n"
        f"Citations:\n" + ("\n".join(citation_lines) if citation_lines else "  (none)") + "\n\n"
        f"Severity rationale: {severity.rationale}\n"
    )

    if len(body) > 2000:
        body = body[: 2000 - 3] + "..."

    return GroundComms(subject=subject, body=body, urgency=urgency)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with degenerate-input guards. Returns 0.0 for zero-variance
    inputs (constant signals) — they are formally undefined."""
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def _infer_subsystem(diagnosis: Diagnosis, context: dict) -> str:
    """Best-effort subsystem inference from context, then matched_failure_modes
    IDs, then primary hypothesis text. Falls back to 'unspecified'."""
    explicit = context.get("subsystem")
    if isinstance(explicit, str) and explicit:
        return explicit

    haystacks: list[str] = []
    haystacks.extend(fm.lower() for fm in diagnosis.matched_failure_modes)
    haystacks.append(diagnosis.primary_hypothesis.lower())
    if "recommended_action" in context and isinstance(context["recommended_action"], str):
        haystacks.append(context["recommended_action"].lower())

    for haystack in haystacks:
        for needle, subsystem in _SUBSYSTEM_INFERENCE_RULES:
            if needle in haystack:
                return subsystem
    return "unspecified"


def _score_to_level(
    score: float,
) -> Literal["info", "warning", "critical", "emergency"]:
    if score < 0.3:
        return "info"
    if score < 0.6:
        return "warning"
    if score < 0.85:
        return "critical"
    return "emergency"


def _severity_to_urgency(level: str) -> Literal["routine", "priority", "immediate"]:
    mapping: dict[str, Literal["routine", "priority", "immediate"]] = {
        "info": "routine",
        "warning": "routine",
        "critical": "priority",
        "emergency": "immediate",
    }
    return mapping.get(level, "routine")


def _action_kind(action: str) -> tuple[int, list[str]]:
    """Map action text to (duration_minutes, required_tools)."""
    a = action.lower()
    if any(k in a for k in ("replace", "swap", "remove and install")):
        return 60, ["spare_part", "torque_wrench"]
    if any(k in a for k in ("reconfigure", "isolate", "reroute")):
        return 45, []
    if any(k in a for k in ("calibrate", "tune", "adjust setpoint")):
        return 30, ["calibration_kit"]
    if any(k in a for k in ("inspect", "verify", "check", "monitor")):
        return 20, []
    return 30, []


def _split_action_into_steps(action: str) -> list[str]:
    """Split a recommended-action string on sentence boundaries; each piece is
    a step. Empty result falls back to a single-step list with the original
    text so the WorkOrder always has at least one step."""
    parts = [p.strip() for p in action.replace(";", ".").split(".") if p.strip()]
    return parts if parts else [action.strip()]
