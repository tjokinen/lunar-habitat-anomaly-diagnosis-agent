"""End-to-end pipeline: telemetry → detectors → agent → event handler."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Sequence

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

# (priority, sequence_number, trigger). Lower priority value pulled first;
# we use -score so higher score = lower priority value = pulled sooner.
# The sequence number breaks ties deterministically (FIFO within a score)
# and prevents the queue from ever needing to compare AnomalyEvent itself.
_QueueItem = tuple[float, int, AnomalyEvent]


async def run_pipeline(
    telemetry_source: TelemetrySource,
    detectors: Sequence[AnomalyDetector],
    agent: ReasoningAgent,
    event_handler: _EventHandler,
    window_size: timedelta = timedelta(hours=1),
    *,
    dedup_cooldown: timedelta = timedelta(minutes=5),
    max_concurrent_investigations: int = 2,
) -> None:
    """Stream telemetry, run detectors, dispatch investigation tasks.

    For each frame:
      1. Emit ``TelemetryFrame`` via *event_handler*.
      2. Feed frame into the agent's ``TelemetryStore``.
      3. Maintain a rolling ``TelemetryWindow`` of length *window_size*.
      4. Run all detectors in parallel on the current window.
      5. Dedup ``AnomalyEvent``s by ``(detector_name, sensor_id)`` within
         *dedup_cooldown*; pass survivors to *event_handler* and enqueue
         them on a priority queue for investigation.

    Investigations are processed by a fixed pool of *max_concurrent_investigations*
    worker tasks pulling from a ``PriorityQueue`` ordered by score (highest first).
    A backed-up low-severity event will never block a freshly-detected high-severity
    one — the worker will pick the new one up on its next ``get()``.
    """
    rolling_frames: list[TelemetryFrame] = []
    # (detector_name, sensor_id) -> last-fired timestamp
    dedup_cache: dict[tuple[str, str], datetime] = {}
    queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
    seq = 0

    async def _emit(event: Event) -> None:
        await event_handler(event)

    workers = [
        asyncio.create_task(_investigation_worker(queue, agent, _emit))
        for _ in range(max_concurrent_investigations)
    ]

    try:
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
                        key = (
                            anomaly_event.detector_name,
                            anomaly_event.affected_sensors[0]
                            if anomaly_event.affected_sensors
                            else "",
                        )
                        last_fired = dedup_cache.get(key)
                        if last_fired is not None:
                            elapsed = (
                                anomaly_event.timestamp - last_fired
                            ).total_seconds()
                            if elapsed < dedup_cooldown.total_seconds():
                                continue
                        dedup_cache[key] = anomaly_event.timestamp

                        await _emit(anomaly_event)

                        # Enqueue for investigation (highest score first).
                        queue.put_nowait(
                            (-anomaly_event.score, seq, anomaly_event)
                        )
                        seq += 1

        # Drain: wait for all queued investigations to finish.
        await queue.join()
    finally:
        # Shut down workers cleanly.
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


async def _investigation_worker(
    queue: asyncio.PriorityQueue[_QueueItem],
    agent: ReasoningAgent,
    emit: _EventHandler,
) -> None:
    """Pull triggers off the priority queue and run the agent on each."""
    while True:
        _, _, trigger = await queue.get()
        try:
            await agent.investigate(trigger, emit)
        except Exception as exc:
            logger.error("Investigation failed for trigger %s: %s", trigger, exc)
        finally:
            queue.task_done()
