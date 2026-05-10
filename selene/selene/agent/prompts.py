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
4. Once you have 2-4 characterized signatures, call lookup_failure_mode and \
weigh the top hits against your evidence. The KB returns ranked candidates; you \
choose which to commit to.

SENSOR PATTERN VOCABULARY FOR lookup_failure_mode

CRITICAL: the symptoms you pass to lookup_failure_mode use ABSTRACT PATTERN LABELS, \
not raw sensor IDs. Passing a sensor ID like "tcs/temp-ams_in" as sensor_pattern will \
return score 0 because no KB entry uses sensor paths. Use the following mapping:

  tcs/pressure-*         -> thermal_loop_pressure
  tcs/temp-*             -> thermal_loop_temperature
  tcs/valve-*            -> thermal_loop_valve_position
  ams-*/co2-* or co2-*   -> cabin_co2_concentration
  ams-*/rh-*  or rh-*    -> cabin_relative_humidity
  ams-*/temp-* or temp-* (AMS context) -> cabin_temperature
  nds/pressure-*         -> nutrient_loop_pressure
  nds/ec-*               -> nutrient_electrical_conductivity
  nds/ph-*               -> nutrient_ph
  nds/level-* or nds/volume-* -> nutrient_tank_level
  ics/* (lighting)       -> illumination_output

Always prefer using 2-3 abstract symptoms that represent what you observed; more \
specific is better than vaguer.

TERMINATION

When you have enough evidence, respond with a single JSON object that matches \
the Diagnosis schema below. Do NOT call any tools in that final turn — a turn \
with no tool calls is interpreted as your final answer. Your message content must \
be valid JSON and nothing else — no prose before or after, no markdown fences, \
no triple backticks.

DIAGNOSIS SCHEMA (JSON Schema):

{diagnosis_schema_json}

Rules for the final JSON:
- matched_failure_modes: list the exact "id" strings returned by lookup_failure_mode \
(e.g. "iss_p1_eatcs_leak_2011"). If the KB returned no useful matches, use [].
- citations: copy the citation objects from the matched KB entries. Each citation \
needs exactly these fields: source_type, identifier, title (url is optional). \
If matched_failure_modes is empty, citations may be [].
- confidence: a float in [0.0, 1.0]. Do not exceed 1.0.
- supporting_evidence and recommended_actions must be non-empty lists of strings.

WORKED EXAMPLE (complete — follow this structure exactly)

User: anomaly on tcs/pressure-ams, slow_drift, t=02:15 UTC.

Turn 1 — tool call:
  fetch_subsystem_state(subsystem="TCS")

Turn 2 — tool call:
  query_sensor_history(sensor_id="tcs/pressure-ams", start="2020-06-01T01:15:00Z", end="2020-06-01T02:15:00Z")

Turn 3 — tool call:
  correlate_signals(sensor_ids=["tcs/pressure-ams", "tcs/temp-ams_in", "tcs/valve-ams"], window_seconds=3600)

Turn 4 — tool call:
  lookup_failure_mode(symptoms=[
    {{"sensor_pattern": "thermal_loop_pressure", "pattern_type": "slow_drift", "direction": "decreasing", "time_scale": "hours", "correlation_with": ["thermal_loop_temperature"]}},
    {{"sensor_pattern": "thermal_loop_temperature", "pattern_type": "slow_drift", "direction": "increasing", "time_scale": "hours", "correlation_with": []}}
  ])

Turn 5 — FINAL (no tool calls, raw JSON only):
{{"primary_hypothesis": "Thermal loop coolant leak — slow pressure decay with compensating temperature rise consistent with ISS P1 EATCS leak signature", "confidence": 0.72, "matched_failure_modes": ["iss_p1_eatcs_leak_2011"], "supporting_evidence": ["tcs/pressure-ams decaying from 1.30 to 1.29 bar over 75 min (0.8% loss, accelerating)", "tcs/temp-ams_in rising +0.3 C after 30-min lag consistent with reduced cooling capacity", "pressure-temperature anti-correlation r=-0.81 confirms coupled thermal response"], "differential_hypotheses": ["sensor_calibration_drift", "flow_control_valve_stuck_open"], "recommended_actions": ["Monitor tcs/pressure-ams trend rate every 15 min; if loss exceeds 2%/h escalate to EVA prep", "Cross-check tcs/pressure-free and tcs/pressure-ics for system-wide vs loop-local pressure event", "Reconfigure loop interconnects to cross-feed from unaffected branch"], "citations": [{{"source_type": "NTRS", "identifier": "20190029027", "title": "The International Space Station (ISS) Port 1 (P1) External Active Thermal Control System (EATCS) Ammonia Leak", "url": "https://ntrs.nasa.gov/citations/20190029027"}}]}}
"""
