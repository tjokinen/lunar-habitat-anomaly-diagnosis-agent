"""Selene — Lunar Habitat Anomaly Diagnosis
Gradio frontend for the Selene pipeline.

Deployment topology
-------------------
The FastAPI backend runs on the AMD droplet (BACKEND_URL / BACKEND_WS_URL).
This Gradio app is deployed as a Hugging Face Space; its Python server
makes the WebSocket + HTTP connections to the droplet server-side, so the
user's browser only talks to Hugging Face over HTTPS.

Steps 3.4–3.7 fill in the panel HTML/JS.  This scaffold (step 3.3) sets up
the layout and CSS so that the page loads with empty panels.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr
from gradio.themes import Base

# ---------------------------------------------------------------------------
# Backend connection settings (overridden by HF Space secrets at deploy time)
# ---------------------------------------------------------------------------

BACKEND_URL    = os.environ.get("BACKEND_URL",    "http://localhost:8000")
BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "ws://localhost:8000")

_RECONNECT_DELAY_SECS = 3.0

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = (Path(__file__).parent / "style.css").read_text()

# ---------------------------------------------------------------------------
# Placeholder HTML for panels not yet populated (steps 3.4–3.7)
# ---------------------------------------------------------------------------

_HABITAT_PLACEHOLDER = """
<div class="selene-panel" style="min-height:420px;display:flex;align-items:center;justify-content:center;">
  <span style="color:#404040;font-size:13px;letter-spacing:0.05em;">
    3D habitat scene — loading…
  </span>
</div>
"""

_DATA_PANEL_PLACEHOLDER = """
<div class="selene-panel">
  <div class="clock-bar">
    <span class="lunar-time" id="lunar-clock">--:--:--</span>
    <span class="comm-window">Earth comm window: —</span>
  </div>
  <table class="sensor-table" id="sensor-table">
    <tr><td colspan="3" style="color:#404040;font-size:12px;padding:20px 8px;">
      Waiting for telemetry…
    </td></tr>
  </table>
</div>
"""

_INVESTIGATION_PLACEHOLDER = """
<div class="selene-panel">
  <div class="trace-header">Investigation trace</div>
  <div id="trace-idle" style="color:#404040;font-size:12px;padding:16px 0;">
    Monitoring nominal — no active investigation.
  </div>
  <div id="trace-active" style="display:none;"></div>
</div>
"""

# ---------------------------------------------------------------------------
# WebSocket → event_state pump
# ---------------------------------------------------------------------------

async def _pump(
    endpoint: str,
    event_type: str,
    payload_key: str,
    queue: asyncio.Queue,
) -> None:
    """Connect to a backend WS endpoint and push messages onto queue forever."""
    import websockets

    url = f"{BACKEND_WS_URL}{endpoint}"
    while True:
        try:
            async with websockets.connect(url) as ws:
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    await queue.put({"type": event_type, payload_key: data})
        except Exception:
            pass
        await asyncio.sleep(_RECONNECT_DELAY_SECS)


async def _event_stream() -> AsyncGenerator[dict, None]:
    """Merge telemetry + agent_event WS streams; yield dicts into event_state."""
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump("/telemetry",    "telemetry",    "frame", queue)),
        asyncio.create_task(_pump("/agent_events", "agent_event",  "event", queue)),
    ]
    try:
        while True:
            item = await queue.get()
            yield item
    finally:
        for t in tasks:
            t.cancel()


# Browser-side MutationObserver that converts event_state mutations into
# custom DOM events dispatched on window.
_OBSERVER_JS = """
() => {
  function attachSeleneObserver() {
    const el = document.querySelector('#event-state');
    if (!el) { setTimeout(attachSeleneObserver, 200); return; }
    const observer = new MutationObserver(() => {
      try {
        const raw = el.querySelector('div') ? el.querySelector('div').textContent : el.textContent;
        const data = JSON.parse(raw || '{}');
        if (!data || !data.type) return;
        if (data.type === 'telemetry') {
          window.dispatchEvent(new CustomEvent('selene:telemetry', { detail: data.frame }));
        } else if (data.type === 'agent_event') {
          window.dispatchEvent(new CustomEvent('selene:agent_event', { detail: data.event }));
        }
      } catch(_) {}
    });
    observer.observe(el, { childList: true, subtree: true, characterData: true });
  }
  attachSeleneObserver();
}
"""

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Selene") as demo:

    # ── Main two-column layout ──────────────────────────────────────────────
    with gr.Row():

        # Left column — 3D habitat + scenario controls
        with gr.Column(scale=3):
            habitat_html = gr.HTML(
                value=_HABITAT_PLACEHOLDER,
                elem_id="habitat-scene",
            )

            # Scenario controls (Gradio components for reliable event wiring)
            with gr.Row(elem_classes=["controls-row"]):
                scenario_dropdown = gr.Dropdown(
                    choices=[],
                    label="Scenario",
                    scale=3,
                    interactive=True,
                )
                start_btn = gr.Button("▶ Start", variant="primary", scale=1)
                reset_btn = gr.Button("↺ Reset", variant="secondary", scale=1)

            status_label = gr.Markdown("**Status:** Idle")

        # Right column — live data panel + investigation trace
        with gr.Column(scale=2):
            data_panel = gr.HTML(
                value=_DATA_PANEL_PLACEHOLDER,
                elem_id="data-panel",
            )
            investigation_panel = gr.HTML(
                value=_INVESTIGATION_PLACEHOLDER,
                elem_id="investigation-panel",
            )

    # Hidden JSON component — the WebSocket pump writes here;
    # browser-side JS watches for mutations and dispatches custom events.
    event_state = gr.JSON(value=None, visible=False, elem_id="event-state")

    # ── Populate scenario dropdown on load ─────────────────────────────────
    async def _load_scenarios() -> gr.Dropdown:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{BACKEND_URL}/scenarios")
                resp.raise_for_status()
                scenarios = resp.json()
                choices = [s["scenario_id"] for s in scenarios]
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        except Exception:
            return gr.Dropdown(choices=[], value=None)

    demo.load(fn=_load_scenarios, inputs=[], outputs=[scenario_dropdown])

    # ── Attach MutationObserver on page load ───────────────────────────────
    demo.load(fn=None, js=_OBSERVER_JS)

    # ── WebSocket event pump — streams backend events into event_state ─────
    demo.load(fn=_event_stream, outputs=[event_state])

    # ── Scenario start / reset ─────────────────────────────────────────────
    async def _start_scenario(scenario_id: str | None) -> str:
        if not scenario_id:
            return "**Status:** No scenario selected"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{BACKEND_URL}/scenario/start",
                    json={"scenario_id": scenario_id},
                )
                resp.raise_for_status()
            return f"**Status:** Running — `{scenario_id}`"
        except Exception as exc:
            return f"**Status:** Error — {exc}"

    async def _reset_scenario() -> str:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{BACKEND_URL}/scenario/reset")
                resp.raise_for_status()
            return "**Status:** Nominal (reset)"
        except Exception as exc:
            return f"**Status:** Error — {exc}"

    start_btn.click(
        fn=_start_scenario,
        inputs=[scenario_dropdown],
        outputs=[status_label],
    )
    reset_btn.click(
        fn=_reset_scenario,
        inputs=[],
        outputs=[status_label],
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
