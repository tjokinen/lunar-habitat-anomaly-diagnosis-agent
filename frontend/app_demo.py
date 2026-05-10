"""LHADA — Lunar Habitat Anomaly Diagnosis Agent (offline public demo).

Self-contained Gradio app for Hugging Face Spaces deployment. Replays a
pre-recorded scenario run from ``demo_recordings/thermal_loop_coolant_leak.json``
and shows the same panels as the live app — habitat scene, sensor table with
sparklines, scenario timeline, investigation trace, final diagnosis card —
without depending on a live FastAPI backend or vLLM endpoint.

The recording was captured by ``selene/scripts/capture_demo_recordings.py``
running against a real backend; every telemetry frame and agent event is
preserved with its original wall-clock offset, so playback timing matches
what a viewer would see in the live system.

Files needed at deploy time
---------------------------
- app_demo.py                                 (this file; HF Space ``app_file``)
- panels.py                                   (HTML/JS panel constants)
- style.css                                   (dark-theme overrides)
- demo_recordings/thermal_loop_coolant_leak.json
- requirements.txt                            (just ``gradio``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr
from gradio.themes import Base

from panels import (
    DATA_PANEL_HTML,
    DATA_PANEL_JS,
    HABITAT_HTML,
    HABITAT_JS,
    INVESTIGATION_HTML,
    INVESTIGATION_JS,
    OBSERVER_JS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("selene.demo")

_HERE = Path(__file__).parent
_CSS = (_HERE / "style.css").read_text()
_RECORDING_PATH = _HERE / "demo_recordings" / "thermal_loop_coolant_leak.json"


# ---------------------------------------------------------------------------
# Recording loader
# ---------------------------------------------------------------------------


def _load_recording() -> dict:
    if not _RECORDING_PATH.exists():
        raise FileNotFoundError(
            f"Demo recording not found at {_RECORDING_PATH}. "
            "Capture one with `selene/scripts/capture_demo_recordings.py`."
        )
    with _RECORDING_PATH.open() as f:
        return json.load(f)


def _compute_effective_speed(recording: dict) -> int:
    """Sim seconds covered per wall second, derived from telemetry timestamps.

    The recording's ``metadata.speed_multiplier`` is informational only —
    in practice it can drift from the *effective* ratio (the original capture
    has a documented case where metadata said 60× but the actual coverage
    was ~480×).  Computing it from the data is the honest number.
    """
    tel = [e for e in recording["events"] if e["channel"] == "telemetry"]
    if len(tel) < 2:
        return 60
    try:
        first = datetime.fromisoformat(
            tel[0]["data"]["timestamp"].replace("Z", "+00:00")
        )
        last = datetime.fromisoformat(
            tel[-1]["data"]["timestamp"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError):
        return 60
    sim_dt = (last - first).total_seconds()
    wall_dt = tel[-1]["t"] - tel[0]["t"]
    if wall_dt <= 0:
        return 60
    return max(1, round(sim_dt / wall_dt))


# Load once at module import — the panels reference values derived from it.
_REC = _load_recording()
_GROUND_TRUTH = _REC["metadata"].get("ground_truth") or {}
_EFFECTIVE_SPEED = _compute_effective_speed(_REC)
_SCENARIO_NAME = "Thermal loop coolant leak (pre-recorded)"
_DURATION = _REC["metadata"].get("duration_seconds", 90.0)
_TOTAL_EVENTS = len(_REC.get("events", []))

logger.info(
    "loaded recording: %d events, effective_speed=%d×, duration=%.1fs",
    _TOTAL_EVENTS, _EFFECTIVE_SPEED, _DURATION,
)


# ---------------------------------------------------------------------------
# Replay logic
# ---------------------------------------------------------------------------


async def _replay_events() -> AsyncGenerator[str, None]:
    """Yield each captured event at its original wall-clock offset, trimmed.

    The leading offset (``events[0]["t"]``) is subtracted so the first event
    fires immediately on Start.  The original capture often has a 10–15s gap
    before the first telemetry frame arrives — backend pipeline startup, not
    part of the scenario story — and replaying that gap looks like the demo
    is broken.

    Each yielded string lands in the hidden ``#event-state`` Textbox; the
    same browser-side ``MutationObserver`` used by the live app dispatches
    ``selene:telemetry`` and ``selene:agent_event`` custom events, so the
    panels behave identically to a live run.
    """
    events = _REC.get("events", [])
    if not events:
        yield ""
        return

    # Minimum gap between yields. Multiple events in the recording often share
    # a wall-time within ~1 ms (e.g. an `agent_run_completed` immediately
    # followed by the next `agent_run_started`), and the browser-side observer
    # polls the hidden textbox at 20 ms intervals. Without this floor, two
    # back-to-back yields can overwrite the textbox before the observer reads
    # it, dropping the earlier event — typically the completion, leaving runs
    # stuck as 'failed' in the trace panel.
    _MIN_GAP = 0.05

    loop = asyncio.get_running_loop()
    base = loop.time()
    leading = float(events[0]["t"])
    last_emit = 0.0
    for seq, ev in enumerate(events, start=1):
        target = base + (float(ev["t"]) - leading)
        # Apply both the original-cadence target AND the minimum-gap floor.
        target = max(target, last_emit + _MIN_GAP)
        delay = target - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if ev["channel"] == "telemetry":
            payload = {"type": "telemetry", "frame": ev["data"], "_seq": seq}
        else:
            payload = {"type": "agent_event", "event": ev["data"], "_seq": seq}
        last_emit = loop.time()
        yield json.dumps(payload)


def _begin_demo() -> tuple[str, str, gr.Button, gr.Button]:
    """Synchronous: flip scenario_state to active, disable Start, enable Reset."""
    payload = {
        "active": True,
        "scenario_id": "thermal_loop_coolant_leak",
        "scenario_name": _SCENARIO_NAME,
        "ground_truth": _GROUND_TRUTH,
    }
    return (
        f"**Status:** Replaying — {_SCENARIO_NAME}",
        json.dumps(payload),
        gr.Button(interactive=False, value="● Replaying…"),
        gr.Button(interactive=True),
    )


def _finish_demo() -> tuple[str, gr.Button]:
    return (
        f"**Status:** Replay complete · {_SCENARIO_NAME}",
        gr.Button(interactive=True, value="▶ Replay again"),
    )


def _reset_demo() -> tuple[str, str, str, gr.Button]:
    """Clear all panels and re-arm the Start button."""
    payload = {"active": False, "scenario_id": None, "ground_truth": None}
    return (
        "**Status:** Ready",
        json.dumps(payload),
        "",                                           # event_state cleared
        gr.Button(interactive=True, value="▶ Start demo"),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_DEMO_BANNER = """\
**Public demo — pre-recorded telemetry replay.** This Space replays a
captured run of the `thermal_loop_coolant_leak` scenario from real EDEN ISS
data with an ISS-inspired anomaly signature injected.  Detection is performed
by DAMP (matrix-profile) on the live system; the diagnosis card is the actual
output of an LLM agent (Qwen 2.5 32B on AMD MI300X) calling typed tools
against a curated NASA-NTRS-cited failure-mode knowledge base — replayed
verbatim here.  Press **Start demo** to play.
"""

with gr.Blocks(title="LHADA — Lunar Habitat Anomaly Diagnosis Agent (Public Demo)") as demo:

    with gr.Row():

        # Left column — 3D habitat + controls + scenario blurb
        with gr.Column(scale=3):
            gr.HTML(value=HABITAT_HTML, elem_id="habitat-scene")

            with gr.Row(elem_classes=["controls-row"]):
                start_btn = gr.Button("▶ Start demo", variant="primary", scale=2)
                reset_btn = gr.Button("↺ Reset", variant="secondary", scale=1, interactive=False)

            gr.Markdown(_DEMO_BANNER, elem_id="demo-banner")
            status_label = gr.Markdown("**Status:** Ready")

        # Right column — live data panel + investigation trace
        with gr.Column(scale=2):
            gr.HTML(value=DATA_PANEL_HTML, elem_id="data-panel")
            gr.HTML(value=INVESTIGATION_HTML, elem_id="investigation-panel")

    # Hidden state textboxes — same wiring as the live app so the existing
    # OBSERVER_JS code finds them and dispatches selene:telemetry /
    # selene:agent_event / selene:scenario events to the panels.
    event_state = gr.Textbox(
        value="",
        elem_id="event-state",
        elem_classes=["selene-hidden"],
        show_label=False,
        lines=1,
        max_lines=1,
        interactive=False,
    )
    scenario_state = gr.Textbox(
        value="",
        elem_id="scenario-state",
        elem_classes=["selene-hidden"],
        show_label=False,
        lines=1,
        max_lines=1,
        interactive=False,
    )

    # ── JS attachments — same set as the live app ───────────────────────
    demo.load(fn=None, js=OBSERVER_JS)
    demo.load(fn=None, js=HABITAT_JS)
    demo.load(fn=None, js=DATA_PANEL_JS)
    demo.load(fn=None, js=INVESTIGATION_JS)

    # Set the speed indicator to the recording's effective ratio so the
    # "Nx" label in the clock bar reflects what the viewer is actually
    # watching, not the (sometimes stale) metadata field.
    demo.load(
        fn=None,
        js=(
            "() => { const apply = () => {"
            "  if (window.__seleneSetSpeed) { "
            f"   window.__seleneSetSpeed({_EFFECTIVE_SPEED}); "
            "    return; "
            "  } "
            "  setTimeout(apply, 200); "
            "}; apply(); }"
        ),
    )

    # ── Click chain ──────────────────────────────────────────────────────
    # 1. _begin_demo  : flip scenario_state to active (drives the timeline)
    # 2. _replay_events: stream captured events to event_state
    # 3. _finish_demo : re-arm the Start button as "Replay again"
    play_chain = (
        start_btn.click(
            fn=_begin_demo,
            outputs=[status_label, scenario_state, start_btn, reset_btn],
        )
        .then(
            fn=_replay_events,
            outputs=[event_state],
        )
        .then(
            fn=_finish_demo,
            outputs=[status_label, start_btn],
        )
    )

    # Reset cancels any in-flight replay so the user can stop early.  The
    # ``cancels`` arg is the supported way to abort a running streaming
    # generator in Gradio 4+/5+.
    reset_btn.click(
        fn=_reset_demo,
        outputs=[status_label, scenario_state, event_state, start_btn],
        cancels=[play_chain],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=Base(),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
