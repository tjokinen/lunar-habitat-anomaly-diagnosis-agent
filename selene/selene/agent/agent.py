"""ReasoningAgent — the LLM-driven investigation loop.

One ``investigate(trigger, emit)`` call corresponds to one diagnostic run. The
loop alternates between LLM completions (with tool schemas exposed) and tool
dispatches; it terminates when the LLM returns a tool-call-free message whose
content parses as a valid ``Diagnosis``. Trace events (``AgentRunStarted``,
``ToolCallStarted``/``Completed``, ``HypothesisLadderUpdated``,
``AgentRunCompleted``/``Failed``) are emitted via the ``emit`` callback so the
frontend can render the agent's reasoning live.

The LLM connection is abstracted behind ``LLMClient`` (a Protocol) so tests can
inject a mock without booting a vLLM endpoint. The default implementation
wraps an ``AsyncOpenAI`` client pointed at the vLLM-served Qwen 2.5 32B from
step 2.5.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from pydantic import ValidationError

logger = logging.getLogger(__name__)

from selene.agent.prompts import TOOL_SCHEMAS, build_system_prompt
from selene.agent.store import TelemetryStore
from selene.agent.tools import (
    correlate_signals,
    fetch_subsystem_state,
    lookup_failure_mode,
    query_sensor_history,
)
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
from selene.core.interfaces import AnomalyEvent, SensorMetadata
from selene.knowledge.models import FailureMode, Signature


# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------


@dataclass
class LLMToolCall:
    """One tool call emitted by the LLM. Arguments are already JSON-decoded."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from one LLM completion. Either ``content`` or ``tool_calls``
    may be present. A response with empty ``tool_calls`` and string ``content``
    is the model's attempt at a final Diagnosis."""
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        force_json: bool,
    ) -> LLMResponse: ...


class _OpenAICompatClient:
    """Default ``LLMClient`` impl. Wraps ``openai.AsyncOpenAI`` against any
    endpoint that speaks the OpenAI chat-completions schema (vLLM, OpenAI itself,
    etc.). ``force_json`` toggles ``response_format={"type": "json_object"}``;
    used on retry after a malformed-JSON final turn."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        # Imported lazily so unit tests don't need openai installed in the env.
        from openai import AsyncOpenAI

        self._base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        logger.info("LLM client → %s  model=%s", base_url, model)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        force_json: bool,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            # vLLM requires --enable-auto-tool-choice server flag for "auto";
            # omitting tool_choice lets the server use its default (same behaviour).
            if not os.environ.get("SELENE_LLM_NO_TOOL_CHOICE"):
                kwargs["tool_choice"] = "auto"
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.warning(
                "LLM completion FAILED  url=%s model=%s err=%s",
                self._base_url, self._model, exc,
            )
            raise
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        message = response.choices[0].message
        n_tool_calls = len(message.tool_calls or [])
        logger.info(
            "LLM completion ok  url=%s model=%s elapsed=%dms tool_calls=%d content=%s",
            self._base_url, self._model, elapsed_ms, n_tool_calls,
            "yes" if message.content else "no",
        )
        tool_calls: list[LLMToolCall] = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                LLMToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )
        return LLMResponse(content=message.content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentTimeoutError(RuntimeError):
    """Raised when the agent exhausts ``max_iterations`` without a valid
    Diagnosis."""


class UnknownToolError(ValueError):
    """Raised when the LLM requests a tool that isn't in the schema. Caught
    inside the loop and reported back to the model via the tool message."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


EmitCallback = Callable[[AgentEvent], Awaitable[None]]


class ReasoningAgent:
    """LLM-driven diagnostic agent.

    The store / KB / metadata are bound at construction time and shared across
    every ``investigate`` call. The LLM client is also bound here, but a fresh
    message history is built per investigation so runs do not bleed into each
    other.
    """

    def __init__(
        self,
        store: TelemetryStore,
        kb: dict[str, FailureMode],
        metadata: SensorMetadata,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_iterations: int = 15,
        client: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.kb = kb
        self.metadata = metadata
        self.max_iterations = max_iterations

        if client is not None:
            self._client: LLMClient = client
        else:
            self._client = _OpenAICompatClient(
                base_url=base_url or os.environ.get(
                    "SELENE_LLM_BASE_URL", "http://localhost:8000/v1"
                ),
                api_key=api_key or os.environ.get("SELENE_LLM_API_KEY", "EMPTY"),
                model=model or os.environ.get(
                    "SELENE_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"
                ),
            )

        self._system_prompt = build_system_prompt()

    async def investigate(
        self,
        trigger: AnomalyEvent,
        emit: EmitCallback,
    ) -> Diagnosis:
        run_id = str(uuid4())
        await emit(
            AgentRunStarted(run_id=run_id, timestamp=_now(), trigger=trigger)
        )

        messages = self._initial_messages(trigger)
        last_turn_was_tool_free = False

        for _ in range(self.max_iterations):
            response = await self._client.complete(
                messages=messages,
                tools=TOOL_SCHEMAS,
                force_json=last_turn_was_tool_free,
            )

            if response.tool_calls:
                messages.append(_assistant_message_with_tool_calls(response))
                for tc in response.tool_calls:
                    await self._execute_and_record(tc, run_id, emit, messages)
                last_turn_was_tool_free = False
                continue

            # Tool-call-free turn: must parse as a valid Diagnosis.
            content = response.content or ""
            messages.append({"role": "assistant", "content": content})

            try:
                diagnosis = Diagnosis.model_validate_json(_strip_markdown(content))
            except ValidationError as e:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the Diagnosis "
                            f"schema. Errors: {e}. Reply with JSON only — no prose, "
                            "no markdown fences."
                        ),
                    }
                )
                last_turn_was_tool_free = True
                continue

            unknown = [
                fm_id
                for fm_id in diagnosis.matched_failure_modes
                if fm_id not in self.kb
            ]
            if unknown:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"matched_failure_modes contains unknown KB IDs: "
                            f"{unknown}. Use only IDs returned by lookup_failure_mode."
                        ),
                    }
                )
                last_turn_was_tool_free = True
                continue

            await emit(
                HypothesisLadderUpdated(
                    run_id=run_id,
                    timestamp=_now(),
                    ranked=[(diagnosis.primary_hypothesis, diagnosis.confidence)],
                )
            )
            await emit(
                AgentRunCompleted(
                    run_id=run_id, timestamp=_now(), diagnosis=diagnosis
                )
            )
            return diagnosis

        await emit(
            AgentRunFailed(run_id=run_id, timestamp=_now(), reason="max_iterations")
        )
        raise AgentTimeoutError(
            f"Agent exhausted {self.max_iterations} iterations without producing "
            "a valid Diagnosis."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_messages(self, trigger: AnomalyEvent) -> list[dict[str, Any]]:
        affected_subsystem = self._infer_affected_subsystem(trigger)
        snapshot_blob: dict[str, Any]
        if affected_subsystem:
            snapshot = self.store.subsystem_state(affected_subsystem, self.metadata)
            snapshot_blob = snapshot.model_dump(mode="json")
        else:
            snapshot_blob = {"note": "no subsystem inferred from trigger sensors"}

        user_payload = {
            "trigger": trigger.model_dump(mode="json"),
            "affected_subsystem": affected_subsystem,
            "available_subsystems": list(self.metadata.subsystems),
            "subsystem_snapshot": snapshot_blob,
        }
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "An anomaly detector has fired. Investigate using the tools "
                    "and produce a Diagnosis. When calling fetch_subsystem_state, "
                    "use a subsystem code from `available_subsystems` (e.g. "
                    f"{', '.join(repr(s) for s in self.metadata.subsystems[:4])}). "
                    "Trigger and current subsystem state:\n"
                    f"{json.dumps(user_payload, indent=2)}"
                ),
            },
        ]

    def _infer_affected_subsystem(self, trigger: AnomalyEvent) -> str | None:
        for sid in trigger.affected_sensors:
            info = self.metadata.sensors.get(sid)
            if info and "subsystem" in info:
                return str(info["subsystem"])
        return None

    async def _execute_and_record(
        self,
        tc: LLMToolCall,
        run_id: str,
        emit: EmitCallback,
        messages: list[dict[str, Any]],
    ) -> None:
        await emit(
            ToolCallStarted(
                run_id=run_id,
                timestamp=_now(),
                call_id=tc.id,
                tool_name=tc.name,
                arguments=tc.arguments,
            )
        )

        summary: str
        result_dict: dict[str, Any]
        error: str | None = None
        try:
            summary, result_dict = await self._dispatch(tc.name, tc.arguments)
        except Exception as exc:  # noqa: BLE001 — we report any failure back to the LLM
            summary = f"error: {exc}"
            result_dict = {"error": str(exc)}
            error = str(exc)

        if tc.name == "lookup_failure_mode" and error is None:
            ranked = result_dict.get("ranked", [])
            await emit(
                HypothesisLadderUpdated(
                    run_id=run_id,
                    timestamp=_now(),
                    ranked=[
                        (entry["failure_mode"]["id"], float(entry["score"]))
                        for entry in ranked
                    ],
                )
            )

        await emit(
            ToolCallCompleted(
                run_id=run_id,
                timestamp=_now(),
                call_id=tc.id,
                tool_name=tc.name,
                result_summary=summary,
                result=result_dict,
                error=error,
            )
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict),
            }
        )

    async def _dispatch(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if name == "query_sensor_history":
            history = await query_sensor_history(
                sensor_id=arguments["sensor_id"],
                start=datetime.fromisoformat(arguments["start"]),
                end=datetime.fromisoformat(arguments["end"]),
                store=self.store,
            )
            return (
                f"{history.sensor_id}: {len(history.points)} points",
                history.model_dump(mode="json"),
            )

        if name == "fetch_subsystem_state":
            snapshot = await fetch_subsystem_state(
                subsystem=arguments["subsystem"],
                store=self.store,
                metadata=self.metadata,
            )
            return (
                f"{snapshot.subsystem}: {len(snapshot.readings)} sensors",
                snapshot.model_dump(mode="json"),
            )

        if name == "correlate_signals":
            report = await correlate_signals(
                sensor_ids=list(arguments["sensor_ids"]),
                window=timedelta(seconds=int(arguments["window_seconds"])),
                store=self.store,
            )
            return (
                f"{len(report.correlations)} pairs over "
                f"{int(arguments['window_seconds'])}s",
                report.model_dump(mode="json"),
            )

        if name == "lookup_failure_mode":
            symptoms = [Signature(**s) for s in arguments["symptoms"]]
            ranked = await lookup_failure_mode(symptoms=symptoms, kb=self.kb)
            result_dict = {
                "ranked": [
                    {
                        "failure_mode": fm.model_dump(mode="json"),
                        "score": float(score),
                    }
                    for fm, score in ranked
                ]
            }
            summary = (
                f"top: {ranked[0][0].id} ({ranked[0][1]:.2f})"
                if ranked
                else "no matches"
            )
            return summary, result_dict

        raise UnknownToolError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Wall-clock UTC. Trace event timestamps are not part of the deterministic
    contract — only tool outputs are."""
    return datetime.now(timezone.utc)


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap JSON in.

    Handles:
      ```json\\n{...}\\n```
      ```\\n{...}\\n```
      {... (already bare JSON — returned as-is)
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        # drop the opening fence line (```json or ```)
        inner = lines[1:]
        # drop the closing ``` if present
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        s = "\n".join(inner).strip()
    return s


def _assistant_message_with_tool_calls(response: LLMResponse) -> dict[str, Any]:
    """Build the OpenAI-format assistant message that records the model's
    tool-call decision so the next turn sees consistent history."""
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ],
    }
