"""Agent-layer type definitions.

Three groups of types:

1. Tool I/O — typed payloads passed between the agent loop and its tools.
2. Agent output — the final ``Diagnosis`` produced when the loop terminates.
3. Agent trace events — broadcast over WebSocket for the frontend; ``type`` is
   the discriminator so downstream consumers can dispatch on a single string.

Pipeline-wide ``Event`` is the top-level union the event handler in step 2.9
will fan out across (telemetry frames, anomaly events, agent trace events).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from selene.core.interfaces import AnomalyEvent, TelemetryFrame
from selene.knowledge.models import Citation


# ---------------------------------------------------------------------------
# Tool I/O
# ---------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float


class TimeSeries(BaseModel):
    sensor_id: str
    unit: str
    points: list[TimeSeriesPoint]  # chronological


class SubsystemSnapshot(BaseModel):
    subsystem: str
    timestamp: datetime
    readings: dict[str, float]  # sensor_id -> latest value
    units: dict[str, str]       # sensor_id -> unit


class SensorCorrelation(BaseModel):
    sensor_a: str
    sensor_b: str
    pearson_r: float = Field(ge=-1.0, le=1.0)
    lag_seconds: float = 0.0  # positive = b lags a


class CorrelationReport(BaseModel):
    sensor_ids: list[str]
    window_start: datetime
    window_end: datetime
    correlations: list[SensorCorrelation]


class SeverityScore(BaseModel):
    level: Literal["info", "warning", "critical", "emergency"]
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class WorkOrder(BaseModel):
    title: str
    subsystem: str
    steps: list[str]
    estimated_duration_minutes: int = Field(ge=0)
    required_tools: list[str] = []


class GroundComms(BaseModel):
    subject: str
    body: str = Field(max_length=2000)
    urgency: Literal["routine", "priority", "immediate"]


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------


class Diagnosis(BaseModel):
    primary_hypothesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_failure_modes: list[str]  # KB entry IDs; agent loop verifies they exist
    supporting_evidence: list[str]
    differential_hypotheses: list[str] = []
    recommended_actions: list[str]
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Agent trace events
# ---------------------------------------------------------------------------


class AgentRunStarted(BaseModel):
    type: Literal["agent_run_started"] = "agent_run_started"
    run_id: str
    timestamp: datetime
    trigger: AnomalyEvent


class ToolCallStarted(BaseModel):
    type: Literal["tool_call_started"] = "tool_call_started"
    run_id: str
    timestamp: datetime
    call_id: str
    tool_name: str
    arguments: dict


class ToolCallCompleted(BaseModel):
    type: Literal["tool_call_completed"] = "tool_call_completed"
    run_id: str
    timestamp: datetime
    call_id: str
    tool_name: str
    result_summary: str  # short string for the trace UI
    result: dict         # full structured result
    error: str | None = None


class HypothesisLadderUpdated(BaseModel):
    type: Literal["hypothesis_ladder_updated"] = "hypothesis_ladder_updated"
    run_id: str
    timestamp: datetime
    ranked: list[tuple[str, float]]  # (failure_mode_id, confidence)


class AgentRunCompleted(BaseModel):
    type: Literal["agent_run_completed"] = "agent_run_completed"
    run_id: str
    timestamp: datetime
    diagnosis: Diagnosis


class AgentRunFailed(BaseModel):
    type: Literal["agent_run_failed"] = "agent_run_failed"
    run_id: str
    timestamp: datetime
    reason: str  # "max_iterations" | "llm_error" | "schema_violation" | ...


# Discriminated union — pydantic v2 uses the ``type`` literal to dispatch.
AgentEvent = Annotated[
    AgentRunStarted
    | ToolCallStarted
    | ToolCallCompleted
    | HypothesisLadderUpdated
    | AgentRunCompleted
    | AgentRunFailed,
    Field(discriminator="type"),
]


# Pipeline-wide event union, consumed by the event handler in step 2.9.
# No discriminator at the top level: TelemetryFrame and AnomalyEvent don't
# carry a ``type`` field. Consumers branch by isinstance() at this layer.
Event = TelemetryFrame | AnomalyEvent | AgentEvent
