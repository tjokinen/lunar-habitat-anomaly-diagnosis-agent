"""End-to-end pipeline: telemetry → detectors → agent → event handler."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from selene.agent.agent import ReasoningAgent
from selene.agent.types import Event
from selene.core.interfaces import (
    AnomalyDetector,
    AnomalyEvent,
    TelemetryFrame,
    TelemetrySource,
    TelemetryWindow,
)

logger = logging.getLogger(__name__)

_EventHandler = Callable[[Event], Awaitable[None]]


async def run_pipeline(
    telemetry_source: TelemetrySource,
    detectors: list[AnomalyDetector],
    agent: ReasoningAgent,
    event_handler: _EventHandler,
    window_size: timedelta = timedelta(hours=1),
    *,
    dedup_cooldown: timedelta = timedelta(minutes=5),
) -> None:
    """Stream telemetry, run detectors, dispatch investigation tasks.

    For each frame:
      1. Emit ``TelemetryFrame`` via *event_handler*.
      2. Feed frame into the agent's ``TelemetryStore``.
      3. Maintain a rolling ``TelemetryWindow`` of length *window_size*.
      4. Run all detectors in parallel on the current window.
      5. Dedup ``AnomalyEvent``s by ``(detector_name, sensor_id)`` within
         *dedup_cooldown*; pass survivors to *event_handler* and launch
         ``agent.investigate`` as a fire-and-forget task.
    """
    rolling_frames: list[TelemetryFrame] = []
    # (detector_name, sensor_id) -> last-fired timestamp
    dedup_cache: dict[tuple[str, str], datetime] = {}
    investigation_tasks: set[asyncio.Task] = set()

    async def _emit(event: Event) -> None:
        await event_handler(event)

    async for frame in telemetry_source.stream():
        # 1. Emit raw telemetry
        await _emit(frame)

        # 2. Feed the agent's store
        agent.store.ingest(frame)

        # 3. Maintain rolling window
        rolling_frames.append(frame)
        cutoff = frame.timestamp - window_size
        rolling_frames = [f for f in rolling_frames if f.timestamp >= cutoff]

        if not rolling_frames:
            continue

        window = TelemetryWindow(
            frames=rolling_frames,
            start=rolling_frames[0].timestamp,
            end=rolling_frames[-1].timestamp,
        )

        # 4. Run detectors in parallel
        if detectors:
            detector_results = await asyncio.gather(
                *(d.evaluate(window) for d in detectors),
                return_exceptions=True,
            )
            for result in detector_results:
                if isinstance(result, BaseException):
                    logger.warning("Detector raised: %s", result)
                    continue
                for anomaly_event in result:
                    # 5. Dedup
                    key = (anomaly_event.detector_name, anomaly_event.affected_sensors[0]
                           if anomaly_event.affected_sensors else "")
                    last_fired = dedup_cache.get(key)
                    if last_fired is not None:
                        elapsed = (anomaly_event.timestamp - last_fired).total_seconds()
                        if elapsed < dedup_cooldown.total_seconds():
                            continue
                    dedup_cache[key] = anomaly_event.timestamp

                    # Emit the deduped anomaly event
                    await _emit(anomaly_event)

                    # Fire-and-forget investigation
                    task = asyncio.create_task(
                        _investigate(agent, anomaly_event, _emit)
                    )
                    investigation_tasks.add(task)
                    task.add_done_callback(investigation_tasks.discard)

    # Wait for any in-flight investigations to finish
    if investigation_tasks:
        await asyncio.gather(*investigation_tasks, return_exceptions=True)


async def _investigate(
    agent: ReasoningAgent,
    trigger: AnomalyEvent,
    emit: _EventHandler,
) -> None:
    try:
        await agent.investigate(trigger, emit)
    except Exception as exc:
        logger.error("Investigation failed for trigger %s: %s", trigger, exc)
