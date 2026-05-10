"""Tests for ReasoningAgent with a mocked LLM client.

Tests use a scripted ``MockLLMClient`` that yields a queue of pre-baked
``LLMResponse`` objects. The agent calls the client; the client pops the next
scripted response and asserts on what was passed in if the script wants to.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from selene.agent.agent import (
    AgentTimeoutError,
    EmitCallback,
    LLMClient,
    LLMResponse,
    LLMToolCall,
    ReasoningAgent,
)
from selene.agent.store import TelemetryStore
from selene.agent.types import (
    AgentEvent,
    AgentRunCompleted,
    AgentRunFailed,
    AgentRunStarted,
    Diagnosis,
    HypothesisLadderUpdated,
    ToolCallCompleted,
    ToolCallStarted,
)
from selene.core.interfaces import (
    AnomalyEvent,
    SensorMetadata,
    SensorReading,
    TelemetryFrame,
)
from selene.knowledge.models import Citation, FailureMode, Signature


_T0 = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
_DT = timedelta(minutes=5)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _build_store() -> TelemetryStore:
    store = TelemetryStore(retention=timedelta(hours=24))
    for i in range(20):
        ts = _T0 + i * _DT
        store.ingest(
            TelemetryFrame(
                timestamp=ts,
                readings={
                    "tcs/loop-press-1": SensorReading(
                        sensor_id="tcs/loop-press-1",
                        timestamp=ts,
                        value=1000.0 - 0.1 * i,
                        unit="kPa",
                    ),
                    "tcs/valve-pos-1": SensorReading(
                        sensor_id="tcs/valve-pos-1",
                        timestamp=ts,
                        value=20.0 + 0.05 * i,
                        unit="%",
                    ),
                },
            )
        )
    return store


def _metadata() -> SensorMetadata:
    return SensorMetadata(
        sensors={
            "tcs/loop-press-1": {"subsystem": "thermal_control_system", "unit": "kPa"},
            "tcs/valve-pos-1": {"subsystem": "thermal_control_system", "unit": "%"},
        },
        subsystems=["thermal_control_system"],
        sampling_rate_seconds=300.0,
    )


def _kb() -> dict[str, FailureMode]:
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
            typical_onset="weeks",
            distinguishing_features=["pressure decay accelerates"],
            differential_diagnosis=[],
            historical_context="...",
            citations=[
                Citation(
                    source_type="NTRS",
                    identifier="20190029027",
                    title="ISS P1 EATCS leak",
                    url=None,
                )
            ],
            typical_response="cross-feed loops",
        ),
    }


def _trigger() -> AnomalyEvent:
    return AnomalyEvent(
        detector_name="mdi",
        timestamp=_T0 + 19 * _DT,
        affected_sensors=["tcs/loop-press-1"],
        score=3.4,
    )


def _good_diagnosis_json() -> str:
    return json.dumps(
        {
            "primary_hypothesis": "Slow ammonia leak in thermal loop A",
            "confidence": 0.78,
            "matched_failure_modes": ["iss_p1_eatcs_leak_2011"],
            "supporting_evidence": ["pressure decayed 0.1 kPa/sample"],
            "differential_hypotheses": [],
            "recommended_actions": ["cross-feed from loop B"],
            "citations": [
                {
                    "source_type": "NTRS",
                    "identifier": "20190029027",
                    "title": "ISS P1 EATCS leak",
                    "url": None,
                }
            ],
        }
    )


# ── Mock LLM client ──────────────────────────────────────────────────────────


class MockLLMClient:
    """Hand-scripted LLM client. Pops one response per ``complete`` call and
    records what was sent for later assertions."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        force_json: bool,
    ) -> LLMResponse:
        if not self.responses:
            raise AssertionError("MockLLMClient ran out of scripted responses")
        self.calls.append(
            {"messages": list(messages), "tools": tools, "force_json": force_json}
        )
        return self.responses.pop(0)


def _agent_with_mock(client: LLMClient, *, max_iterations: int = 5) -> ReasoningAgent:
    return ReasoningAgent(
        store=_build_store(),
        kb=_kb(),
        metadata=_metadata(),
        client=client,
        max_iterations=max_iterations,
    )


def _collect_events() -> tuple[list[AgentEvent], EmitCallback]:
    """Returns (events list, async emit callable)."""
    events: list[AgentEvent] = []

    async def emit(ev: AgentEvent) -> None:
        events.append(ev)

    return events, emit


# ── Happy path ───────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_three_tool_calls_then_diagnosis(self):
        """Agent calls query_sensor_history → correlate_signals → lookup_failure_mode
        → emits a Diagnosis. Verify all five trace event types appear in order."""
        responses = [
            # Turn 1: query_sensor_history
            LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        id="tc-1",
                        name="query_sensor_history",
                        arguments={
                            "sensor_id": "tcs/loop-press-1",
                            "start": (_T0).isoformat(),
                            "end": (_T0 + 19 * _DT).isoformat(),
                        },
                    )
                ]
            ),
            # Turn 2: correlate_signals
            LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        id="tc-2",
                        name="correlate_signals",
                        arguments={
                            "sensor_ids": ["tcs/loop-press-1", "tcs/valve-pos-1"],
                            "window_seconds": 3600,
                        },
                    )
                ]
            ),
            # Turn 3: lookup_failure_mode
            LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        id="tc-3",
                        name="lookup_failure_mode",
                        arguments={
                            "symptoms": [
                                {
                                    "sensor_pattern": "thermal_loop_pressure",
                                    "pattern_type": "slow_drift",
                                    "direction": "decreasing",
                                    "time_scale": "weeks",
                                }
                            ]
                        },
                    )
                ]
            ),
            # Turn 4: final Diagnosis
            LLMResponse(content=_good_diagnosis_json()),
        ]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        diagnosis = asyncio.run(agent.investigate(_trigger(), emit))

        assert isinstance(diagnosis, Diagnosis)
        assert diagnosis.matched_failure_modes == ["iss_p1_eatcs_leak_2011"]

        # Event ordering check
        types = [type(ev).__name__ for ev in events]
        assert types[0] == "AgentRunStarted"
        assert types[-1] == "AgentRunCompleted"

        # All five event types should appear
        present = set(types)
        assert "AgentRunStarted" in present
        assert "ToolCallStarted" in present
        assert "ToolCallCompleted" in present
        assert "HypothesisLadderUpdated" in present
        assert "AgentRunCompleted" in present

        # Three tools dispatched → three Started/Completed pairs
        started = [ev for ev in events if isinstance(ev, ToolCallStarted)]
        completed = [ev for ev in events if isinstance(ev, ToolCallCompleted)]
        assert len(started) == 3
        assert len(completed) == 3
        assert [s.tool_name for s in started] == [
            "query_sensor_history",
            "correlate_signals",
            "lookup_failure_mode",
        ]

        # HypothesisLadderUpdated fires at least once after lookup_failure_mode
        ladders = [ev for ev in events if isinstance(ev, HypothesisLadderUpdated)]
        assert ladders
        assert ladders[0].ranked[0][0] == "iss_p1_eatcs_leak_2011"

    def test_first_call_carries_subsystem_snapshot(self):
        """Initial user message should include the affected subsystem snapshot."""
        client = MockLLMClient([LLMResponse(content=_good_diagnosis_json())])
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        asyncio.run(agent.investigate(_trigger(), emit))

        first_messages = client.calls[0]["messages"]
        user_msg = next(m for m in first_messages if m["role"] == "user")
        assert "thermal_control_system" in user_msg["content"]
        assert "tcs/loop-press-1" in user_msg["content"]


# ── Tool error handling ──────────────────────────────────────────────────────


class TestToolErrorHandling:
    def test_tool_error_recorded_and_loop_continues(self):
        """When a tool raises, ToolCallCompleted.error is set and the next
        iteration runs normally."""
        responses = [
            # Invalid arguments → tool dispatch will fail
            LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        id="tc-bad",
                        name="query_sensor_history",
                        arguments={
                            "sensor_id": "tcs/loop-press-1",
                            "start": "not-a-date",
                            "end": "still-not-a-date",
                        },
                    )
                ]
            ),
            # Recovery: emit the diagnosis
            LLMResponse(content=_good_diagnosis_json()),
        ]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        diagnosis = asyncio.run(agent.investigate(_trigger(), emit))
        assert isinstance(diagnosis, Diagnosis)

        completed = [ev for ev in events if isinstance(ev, ToolCallCompleted)]
        assert len(completed) == 1
        assert completed[0].error is not None
        assert "error" in completed[0].result

    def test_unknown_tool_reported_as_error(self):
        """LLM hallucinating a tool name lands in the error path, not a crash."""
        responses = [
            LLMResponse(
                tool_calls=[
                    LLMToolCall(id="tc-x", name="nonexistent_tool", arguments={})
                ]
            ),
            LLMResponse(content=_good_diagnosis_json()),
        ]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        diagnosis = asyncio.run(agent.investigate(_trigger(), emit))
        assert isinstance(diagnosis, Diagnosis)

        completed = [ev for ev in events if isinstance(ev, ToolCallCompleted)]
        assert completed[0].error is not None
        assert "nonexistent_tool" in completed[0].error


# ── JSON retry path ──────────────────────────────────────────────────────────


class TestJSONRetry:
    def test_malformed_json_then_correction(self):
        """When the model returns invalid JSON in a tool-call-free turn, the
        agent appends a corrective message and retries with force_json=True."""
        responses = [
            LLMResponse(content="here is my answer: leak in thermal loop"),
            LLMResponse(content=_good_diagnosis_json()),
        ]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        diagnosis = asyncio.run(agent.investigate(_trigger(), emit))
        assert isinstance(diagnosis, Diagnosis)

        # Second LLM call should have force_json set
        assert client.calls[1]["force_json"] is True
        # First call should not
        assert client.calls[0]["force_json"] is False

        # The corrective message should be in the history
        retry_messages = client.calls[1]["messages"]
        assert any(
            m["role"] == "user" and "Diagnosis schema" in m["content"]
            for m in retry_messages
        )

    def test_rejects_unknown_failure_mode_id(self):
        """If matched_failure_modes references a KB ID that doesn't exist,
        the agent rejects and asks the model to fix it."""
        bogus = json.dumps(
            {
                "primary_hypothesis": "x",
                "confidence": 0.5,
                "matched_failure_modes": ["does_not_exist"],
                "supporting_evidence": [],
                "recommended_actions": [],
                "citations": [],
            }
        )
        responses = [
            LLMResponse(content=bogus),
            LLMResponse(content=_good_diagnosis_json()),
        ]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        diagnosis = asyncio.run(agent.investigate(_trigger(), emit))
        assert diagnosis.matched_failure_modes == ["iss_p1_eatcs_leak_2011"]

        retry_messages = client.calls[1]["messages"]
        assert any(
            m["role"] == "user" and "unknown KB IDs" in m["content"]
            for m in retry_messages
        )


# ── Max iterations ───────────────────────────────────────────────────────────


class TestMaxIterations:
    def test_exhausts_iterations_and_emits_failure(self):
        """If the model never produces a valid Diagnosis, the agent emits
        AgentRunFailed and raises AgentTimeoutError."""
        # Always return malformed JSON
        responses = [LLMResponse(content="nope") for _ in range(5)]
        client = MockLLMClient(responses)
        agent = _agent_with_mock(client, max_iterations=3)
        events, emit = _collect_events()

        with pytest.raises(AgentTimeoutError):
            asyncio.run(agent.investigate(_trigger(), emit))

        failed = [ev for ev in events if isinstance(ev, AgentRunFailed)]
        assert len(failed) == 1
        assert failed[0].reason == "max_iterations"

        # Exactly max_iterations LLM calls were made
        assert len(client.calls) == 3


# ── Initial state ────────────────────────────────────────────────────────────


class TestInitialState:
    def test_run_started_event_carries_trigger(self):
        client = MockLLMClient([LLMResponse(content=_good_diagnosis_json())])
        agent = _agent_with_mock(client)
        events, emit = _collect_events()

        asyncio.run(agent.investigate(_trigger(), emit))

        started = [ev for ev in events if isinstance(ev, AgentRunStarted)]
        assert len(started) == 1
        assert started[0].trigger.detector_name == "mdi"
        assert started[0].run_id == [
            ev for ev in events if isinstance(ev, AgentRunCompleted)
        ][0].run_id

    def test_constructor_reads_env_vars(self, monkeypatch):
        """If no client is injected, env vars drive the OpenAI client config.
        This test only verifies the env-var read path; it does not hit the
        network. Construction must not raise even if vLLM is not running."""
        monkeypatch.setenv("SELENE_LLM_BASE_URL", "http://example.test/v1")
        monkeypatch.setenv("SELENE_LLM_API_KEY", "test-key")
        monkeypatch.setenv("SELENE_LLM_MODEL", "test-model")
        agent = ReasoningAgent(
            store=_build_store(), kb=_kb(), metadata=_metadata()
        )
        # The internal client was built without raising
        assert agent._client is not None
