"""FastAPI backend for the Selene pipeline.

Exposes:
  WS  /telemetry      — streams TelemetryFrame JSON to subscribers
  WS  /agent_events   — streams AgentEvent JSON to subscribers
  POST /scenario/start  {"scenario_id": str}
  POST /scenario/reset
  GET  /sensors
  GET  /scenarios
  GET  /knowledge
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from selene.agent.agent import ReasoningAgent
from selene.agent.store import TelemetryStore
from selene.agent.types import AgentEvent, Event
from selene.core.interfaces import AnomalyEvent, TelemetryFrame
from selene.data.injector import ScenarioInjector
from selene.data.replayer import EdenIssReplayer
from selene.detection.damp import DampDetector
from selene.detection.threshold import ThresholdDetector
from selene.knowledge.loader import load_kb
from selene.pipeline import run_pipeline
from selene.scenarios.loader import load_scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global application state (populated at startup / scenario changes)
# ---------------------------------------------------------------------------

_DATA_PATH: Path = Path("data/eden_iss/edeniss2020")
_SPEED_MULTIPLIER: float = 60.0

_scenario_registry: dict[str, Any] = {}   # scenario_id -> AnomalyModule
_active_scenario_id: str | None = None
_pipeline_task: asyncio.Task | None = None
_pipeline_stop: asyncio.Event = asyncio.Event()

_telemetry_subs: set[WebSocket] = set()
_agent_event_subs: set[WebSocket] = set()

_store: TelemetryStore | None = None
_agent: ReasoningAgent | None = None
_replayer: EdenIssReplayer | None = None


# ---------------------------------------------------------------------------
# Pub/sub broadcaster helpers
# ---------------------------------------------------------------------------

async def _broadcast(sockets: set[WebSocket], payload: str) -> None:
    dead: set[WebSocket] = set()
    for ws in list(sockets):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    sockets -= dead


async def _event_handler(event: Event) -> None:
    if isinstance(event, TelemetryFrame):
        await _broadcast(_telemetry_subs, event.model_dump_json())
    elif isinstance(event, AnomalyEvent):
        # Also send anomaly events over the agent_events channel so the frontend
        # can show detector firings in the investigation panel.
        await _broadcast(_agent_event_subs, event.model_dump_json())
    else:
        # AgentEvent variants — all have model_dump_json()
        await _broadcast(_agent_event_subs, event.model_dump_json())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

def _build_detectors(replayer: EdenIssReplayer) -> list:
    meta = replayer.get_metadata()
    config_path = Path(__file__).parent.parent.parent / "config" / "sensor_ranges.yaml"

    detectors = []

    # Threshold detector (always on — low overhead)
    if config_path.exists():
        detectors.append(ThresholdDetector.from_yaml(config_path, metadata=meta))

    # DAMP detector on TCS pressure and temperature sensors
    tcs_sensors = [
        sid for sid, info in meta.sensors.items()
        if info.get("subsystem") == "TCS" and info.get("sensor_type") in ("P", "T")
    ]
    if tcs_sensors:
        detectors.append(DampDetector(
            sensor_ids=tcs_sensors[:6],   # keep it manageable; top 6 TCS P/T sensors
            window_length=12,             # 1 hour at 5-min cadence
            threshold=2.0,
        ))

    return detectors


async def _run_pipeline_task(
    data_path: Path,
    scenario_id: str | None,
    speed_multiplier: float = 60.0,
) -> None:
    global _store, _agent, _replayer

    # When a scenario is selected, fast-forward the replayer to a short
    # warmup window before the anomaly onset. Otherwise the replayer would
    # tick through months of nominal data (the dataset starts in early 2020,
    # the scenarios are anchored at 2020-06-01) before the anomaly window.
    replayer_start = None
    if scenario_id and scenario_id in _scenario_registry:
        try:
            gt = _scenario_registry[scenario_id].get_ground_truth()
            replayer_start = gt.start_time - timedelta(minutes=30)
        except Exception as exc:
            logger.warning("Could not resolve scenario start_time: %s", exc)

    replayer = EdenIssReplayer(
        data_path,
        start_time=replayer_start,
        speed_multiplier=speed_multiplier,
    )
    _replayer = replayer

    source = replayer
    if scenario_id and scenario_id in _scenario_registry:
        module = _scenario_registry[scenario_id]
        source = ScenarioInjector(replayer, [module])  # type: ignore[arg-type]

    kb = load_kb()
    store = TelemetryStore(retention=timedelta(hours=2))
    _store = store

    agent = ReasoningAgent(store=store, kb=kb, metadata=replayer.get_metadata())
    _agent = agent

    detectors = _build_detectors(replayer)

    try:
        await run_pipeline(
            telemetry_source=source,  # type: ignore[arg-type]
            detectors=detectors,
            agent=agent,
            event_handler=_event_handler,
        )
    except asyncio.CancelledError:
        logger.info("Pipeline task cancelled")
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)


async def _start_pipeline(scenario_id: str | None = None) -> None:
    global _pipeline_task, _active_scenario_id
    if _pipeline_task and not _pipeline_task.done():
        _pipeline_task.cancel()
        try:
            await _pipeline_task
        except (asyncio.CancelledError, Exception):
            pass
    _active_scenario_id = scenario_id
    _pipeline_task = asyncio.create_task(
        _run_pipeline_task(_DATA_PATH, scenario_id, _SPEED_MULTIPLIER)
    )


def _scenario_active_payload() -> dict[str, Any]:
    """Return the current scenario + its ground-truth window (or {active:false})."""
    if _active_scenario_id is None or _active_scenario_id not in _scenario_registry:
        return {"active": False, "scenario_id": None, "ground_truth": None}
    module = _scenario_registry[_active_scenario_id]
    try:
        gt = module.get_ground_truth().model_dump(mode="json")
    except Exception:
        gt = None
    return {
        "active": True,
        "scenario_id": _active_scenario_id,
        "ground_truth": gt,
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Scan scenarios/ directory at startup
    scenarios_dir = Path(__file__).parent.parent.parent / "scenarios"
    if scenarios_dir.is_dir():
        for yaml_path in sorted(scenarios_dir.glob("*.yaml")):
            scenario_id = yaml_path.stem
            try:
                module = load_scenario(yaml_path)
                _scenario_registry[scenario_id] = module
                logger.info("Registered scenario: %s", scenario_id)
            except Exception as exc:
                logger.warning("Failed to load scenario %s: %s", yaml_path.name, exc)

    # Start nominal pipeline (no scenario)
    await _start_pipeline(scenario_id=None)

    yield

    # Shutdown
    if _pipeline_task and not _pipeline_task.done():
        _pipeline_task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Selene — Lunar Habitat Anomaly Diagnosis", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class ScenarioStartRequest(BaseModel):
    scenario_id: str


class SpeedRequest(BaseModel):
    multiplier: float | None  # None = as-fast-as-possible


@app.get("/sensors")
async def get_sensors():
    if _replayer is None:
        raise HTTPException(status_code=503, detail="Pipeline not started")
    return _replayer.get_metadata()


@app.get("/scenarios")
async def get_scenarios():
    return [
        {
            "scenario_id": sid,
            "name": module.name,
            "description": module.description,
            "affected_sensors": module.affected_sensors,
        }
        for sid, module in _scenario_registry.items()
    ]


@app.get("/knowledge")
async def get_knowledge():
    kb = load_kb()
    return [
        {
            "id": fm.id,
            "name": fm.name,
            "affected_subsystems": fm.affected_subsystems,
            "summary": fm.typical_onset,
        }
        for fm in kb.values()
    ]


@app.post("/scenario/start")
async def start_scenario(req: ScenarioStartRequest):
    if req.scenario_id not in _scenario_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id {req.scenario_id!r}. "
                   f"Known: {sorted(_scenario_registry)}",
        )
    await _start_pipeline(scenario_id=req.scenario_id)
    return {
        **_scenario_active_payload(),
        "status": "started",
        "scenario_id": req.scenario_id,
    }


@app.post("/scenario/reset")
async def reset_scenario():
    await _start_pipeline(scenario_id=None)
    return {**_scenario_active_payload(), "status": "reset"}


@app.get("/scenario/active")
async def get_active_scenario():
    """Current scenario + its ground-truth window. Used by the frontend to
    render the anomaly timeline (when did it start, when does it end)."""
    return _scenario_active_payload()


@app.get("/replay/speed")
async def get_replay_speed():
    """Current replay speed multiplier (default applied to fresh pipelines)."""
    return {
        "multiplier": (
            _replayer.get_speed() if _replayer is not None else _SPEED_MULTIPLIER
        ),
        "default": _SPEED_MULTIPLIER,
    }


@app.post("/replay/speed")
async def set_replay_speed(req: SpeedRequest):
    """Update replay speed live and as the default for future pipelines."""
    global _SPEED_MULTIPLIER
    _SPEED_MULTIPLIER = req.multiplier if req.multiplier is not None else 60.0
    if _replayer is not None:
        _replayer.set_speed(req.multiplier)
    return {"status": "ok", "multiplier": req.multiplier}


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    _telemetry_subs.add(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep alive; client sends nothing
    except WebSocketDisconnect:
        pass
    finally:
        _telemetry_subs.discard(websocket)


@app.websocket("/agent_events")
async def ws_agent_events(websocket: WebSocket):
    await websocket.accept()
    _agent_event_subs.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _agent_event_subs.discard(websocket)
