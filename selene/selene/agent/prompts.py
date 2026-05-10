"""System prompt and OpenAI tool schemas for ``ReasoningAgent``.

The schemas mirror the LLM-facing parameters of the tools in ``selene/agent/tools.py``
— store/kb/metadata are bound by the agent at construction time and are *not*
exposed to the LLM.

The system prompt describes the role, the tool palette, the termination rule
(emit a tool-call-free message whose content parses as ``Diagnosis``), and a
worked example so the model commits to the JSON-final convention.
"""

from __future__ import annotations

from selene.agent.types import Diagnosis


def build_system_prompt() -> str:
    """Return the canonical system prompt. Pure function — no env reads."""
    schema = Diagnosis.model_json_schema()
    return _SYSTEM_PROMPT_TEMPLATE.format(diagnosis_schema_json=schema)


# Tool schemas in OpenAI function-calling format. Names match the function
# names in ``selene/agent/tools.py`` 1:1 — the agent dispatches by name.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_sensor_history",
            "description": (
                "Fetch the chronologically ordered readings for one sensor over an "
                "ISO-8601 time window. Returns an empty series for unknown sensors. "
                "Use this to inspect a single signal in detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor_id": {
                        "type": "string",
                        "description": "Sensor ID, e.g. 'tcs/loop-press-1'.",
                    },
                    "start": {
                        "type": "string",
                        "description": "ISO-8601 start timestamp (inclusive).",
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO-8601 end timestamp (inclusive).",
                    },
                },
                "required": ["sensor_id", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_subsystem_state",
            "description": (
                "Latest reading per sensor in a subsystem, anchored at the most "
                "recent timestamp across those sensors. Use to get a system-wide "
                "snapshot before zooming in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "description": (
                            "Subsystem name, e.g. 'thermal_control_system', "
                            "'atmosphere_management_system', "
                            "'nutrient_delivery_system', 'illumination_control_system'."
                        ),
                    }
                },
                "required": ["subsystem"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlate_signals",
            "description": (
                "Pairwise zero-lag Pearson correlation across the listed sensors over "
                "the most recent ``window_seconds`` of telemetry. Use to test whether "
                "sensors that should be coupled (e.g. loop pressure vs. valve position) "
                "are still moving together."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "description": "Sensors to correlate. At least 2.",
                    },
                    "window_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Width of the analysis window in seconds, ending at "
                            "the most recent reading."
                        ),
                    },
                },
                "required": ["sensor_ids", "window_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_failure_mode",
            "description": (
                "Rank knowledge-base failure modes by symptom overlap. Pass a list "
                "of observed signatures (sensor pattern, pattern type, direction, "
                "time scale). Use after you have characterized the signal shape."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sensor_pattern": {"type": "string"},
                                "pattern_type": {
                                    "type": "string",
                                    "enum": [
                                        "slow_drift",
                                        "step_change",
                                        "oscillation",
                                        "threshold_breach",
                                    ],
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": ["increasing", "decreasing", "either"],
                                },
                                "time_scale": {
                                    "type": "string",
                                    "enum": ["minutes", "hours", "days", "weeks"],
                                },
                                "correlation_with": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                            },
                            "required": [
                                "sensor_pattern",
                                "pattern_type",
                                "direction",
                                "time_scale",
                            ],
                        },
                        "minItems": 1,
                    }
                },
                "required": ["symptoms"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE = """You are Selene, a life-support telemetry investigator on a lunar habitat. \
Earth-comm round-trip latency means crew cannot wait for ground-loop diagnosis on \
fast-moving anomalies. Your job is to reason from telemetry to a structured Diagnosis \
with citations to the failure-mode knowledge base.

You have four read-only tools:

- query_sensor_history(sensor_id, start, end) → time series for one sensor.
- fetch_subsystem_state(subsystem) → latest reading per sensor in a subsystem.
- correlate_signals(sensor_ids, window_seconds) → pairwise zero-lag Pearson r over \
the most recent window. Use this to verify expected couplings (e.g. loop pressure ↔ \
valve position) — broken couplings are diagnostic.
- lookup_failure_mode(symptoms) → KB entries ranked by symptom overlap. Call this \
*after* you have characterized the signal shape, not before.

INVESTIGATIVE STYLE

1. Start by fetching the affected subsystem's state to orient yourself.
2. For each suspect sensor, pull a history window and characterize the shape \
(slow drift / step change / oscillation / threshold breach) and direction.
3. Test couplings between sensors that should move together. A sensor moving alone \
is more diagnostic than one moving in concert.
4. Once you have 2–4 characterized signatures, call lookup_failure_mode and \
weigh the top hits against your evidence. The KB returns ranked candidates; you \
choose which to commit to.

TERMINATION

When you have enough evidence, respond with a single JSON object that matches \
the Diagnosis schema below. Do NOT call any tools in that final turn — a turn \
with no tool calls is interpreted as your final answer. Your message content must \
be valid JSON, nothing else (no prose, no markdown fences).

DIAGNOSIS SCHEMA (JSON Schema):

{diagnosis_schema_json}

The ``matched_failure_modes`` field MUST contain only KB entry IDs that the \
lookup_failure_mode tool returned. ``citations`` should be drawn from those entries' \
citation lists. ``confidence`` reflects your subjective certainty in [0, 1] — be \
honest, not optimistic.

WORKED EXAMPLE (illustrative, not a script)

User: anomaly on tcs/loop-press-1, slow_drift score 3.4, t=12:00 UTC.

Turn 1: call fetch_subsystem_state(subsystem="thermal_control_system").
Turn 2: call query_sensor_history(sensor_id="tcs/loop-press-1", start=..., end=...).
Turn 3: call correlate_signals(sensor_ids=["tcs/loop-press-1", "tcs/valve-pos-1"], \
window_seconds=3600).
Turn 4: call lookup_failure_mode with two characterized symptoms.
Turn 5 (final, no tool calls): emit JSON Diagnosis with primary_hypothesis, \
matched_failure_modes from the KB hit, supporting_evidence citing the actual values \
seen, and citations from the KB entry.
"""
