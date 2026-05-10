"""Tests for the FastAPI backend.

Uses Starlette's TestClient (which runs the lifespan and thus populates
the scenario registry) for REST tests, and httpx AsyncClient for async
tests that don't need the lifespan.  WebSocket tests use TestClient's
ws_connect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

import selene.api.app as api_app
from selene.api.app import app

FIXTURE = Path(__file__).parent.parent / "fixtures" / "eden_iss_sample"


def _patch_pipeline():
    """Patch _start_pipeline so tests don't spin up the real replayer."""
    return patch.object(api_app, "_start_pipeline", new_callable=AsyncMock)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient with lifespan — populates scenario registry."""
    with _patch_pipeline():
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_with_replayer():
    """TestClient with lifespan + a real replayer attached (for /sensors)."""
    from selene.data.replayer import EdenIssReplayer
    replayer = EdenIssReplayer(FIXTURE, speed_multiplier=None)
    with _patch_pipeline():
        api_app._replayer = replayer
        with TestClient(app) as c:
            yield c
    api_app._replayer = None


# ---------------------------------------------------------------------------
# GET /scenarios
# ---------------------------------------------------------------------------

class TestGetScenarios:
    def test_returns_list(self, client: TestClient):
        resp = client.get("/scenarios")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_known_scenarios_present(self, client: TestClient):
        ids = [s["scenario_id"] for s in client.get("/scenarios").json()]
        assert "test_step_change" in ids
        assert "thermal_loop_coolant_leak" in ids

    def test_scenario_has_required_fields(self, client: TestClient):
        for s in client.get("/scenarios").json():
            assert "scenario_id" in s
            assert "name" in s
            assert "description" in s
            assert "affected_sensors" in s


# ---------------------------------------------------------------------------
# GET /knowledge
# ---------------------------------------------------------------------------

class TestGetKnowledge:
    def test_returns_five_entries(self, client: TestClient):
        resp = client.get("/knowledge")
        assert resp.status_code == 200
        assert len(resp.json()) == 5

    def test_entry_has_required_fields(self, client: TestClient):
        for entry in client.get("/knowledge").json():
            assert "id" in entry
            assert "name" in entry
            assert "affected_subsystems" in entry


# ---------------------------------------------------------------------------
# GET /sensors
# ---------------------------------------------------------------------------

class TestGetSensors:
    def test_503_when_pipeline_not_started(self, client: TestClient):
        original = api_app._replayer
        api_app._replayer = None
        try:
            resp = client.get("/sensors")
            assert resp.status_code == 503
        finally:
            api_app._replayer = original

    def test_returns_sensor_metadata_when_ready(self, client_with_replayer: TestClient):
        resp = client_with_replayer.get("/sensors")
        assert resp.status_code == 200
        data = resp.json()
        assert "sensors" in data
        assert "subsystems" in data
        assert "sampling_rate_seconds" in data

    def test_sensors_include_tcs(self, client_with_replayer: TestClient):
        data = client_with_replayer.get("/sensors").json()
        assert "TCS" in data["subsystems"]


# ---------------------------------------------------------------------------
# POST /scenario/start  and  POST /scenario/reset
# ---------------------------------------------------------------------------

class TestScenarioLifecycle:
    def test_unknown_scenario_404(self, client: TestClient):
        resp = client.post("/scenario/start", json={"scenario_id": "does_not_exist"})
        assert resp.status_code == 404
        assert "does_not_exist" in resp.json()["detail"]

    def test_known_scenario_200(self, client: TestClient):
        with _patch_pipeline() as mock_start:
            resp = client.post(
                "/scenario/start",
                json={"scenario_id": "thermal_loop_coolant_leak"},
            )
        assert resp.status_code == 200
        assert resp.json()["scenario_id"] == "thermal_loop_coolant_leak"
        mock_start.assert_called_once_with(scenario_id="thermal_loop_coolant_leak")

    def test_reset_returns_200(self, client: TestClient):
        with _patch_pipeline() as mock_start:
            resp = client.post("/scenario/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"
        mock_start.assert_called_once_with(scenario_id=None)


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

class TestWebSockets:
    def test_telemetry_ws_accepts_connection(self, client: TestClient):
        with client.websocket_connect("/telemetry") as ws:
            assert ws is not None

    def test_agent_events_ws_accepts_connection(self, client: TestClient):
        with client.websocket_connect("/agent_events") as ws:
            assert ws is not None

    def test_telemetry_ws_receives_broadcast(self, client: TestClient):
        from datetime import datetime, timezone
        from selene.core.interfaces import SensorReading, TelemetryFrame

        ts = datetime(2020, 6, 1, tzinfo=timezone.utc)
        frame = TelemetryFrame(
            timestamp=ts,
            readings={
                "tcs/pressure-ams": SensorReading(
                    sensor_id="tcs/pressure-ams",
                    timestamp=ts,
                    value=1.0,
                    unit="bar",
                )
            },
        )
        with client.websocket_connect("/telemetry") as ws:
            # Trigger broadcast via the event handler (runs in the TestClient's
            # event loop, so asyncio.run is safe here)
            asyncio.run(
                api_app._broadcast(api_app._telemetry_subs, frame.model_dump_json())
            )
            data = json.loads(ws.receive_text())
            assert "timestamp" in data
            assert "readings" in data

    def test_agent_events_ws_receives_broadcast(self, client: TestClient):
        from datetime import datetime, timezone
        from selene.core.interfaces import AnomalyEvent

        event = AnomalyEvent(
            detector_name="threshold",
            timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
            affected_sensors=["tcs/pressure-ams"],
            score=0.5,
        )
        with client.websocket_connect("/agent_events") as ws:
            asyncio.run(
                api_app._broadcast(
                    api_app._agent_event_subs, event.model_dump_json()
                )
            )
            data = json.loads(ws.receive_text())
            assert data["detector_name"] == "threshold"
            assert data["score"] == pytest.approx(0.5)
