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

_HABITAT_HTML = """
<div id="selene-habitat-root" style="width:100%;height:440px;position:relative;background:#050508;border-radius:4px;overflow:hidden;">
  <canvas id="selene-habitat-canvas" style="width:100%;height:100%;display:block;"></canvas>
  <!-- subsystem legend -->
  <div id="selene-legend" style="position:absolute;bottom:10px;left:12px;font-family:monospace;font-size:10px;display:flex;gap:12px;pointer-events:none;">
    <span><span id="leg-thermal"  style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#60a5fa;margin-right:4px;"></span>THERMAL</span>
    <span><span id="leg-atmos"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;margin-right:4px;"></span>ATMOS</span>
    <span><span id="leg-nutrient" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#facc15;margin-right:4px;"></span>NUTRIENTS</span>
    <span><span id="leg-illum"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f97316;margin-right:4px;"></span>ILLUMIN</span>
  </div>
  <div id="selene-earth-label" style="position:absolute;top:10px;right:12px;font-family:monospace;font-size:10px;color:#475569;pointer-events:none;">
    Earth — comm delay: 2.6 s
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {
  'use strict';

  // ── guard against double-init on Gradio HMR ──────────────────────────────
  if (window._seleneHabitatInit) return;
  window._seleneHabitatInit = true;

  // ── subsystem palette ─────────────────────────────────────────────────────
  const SUBSYSTEMS = {
    thermal:  { color: 0x1d4ed8, emissive: 0x1e3a5f, label: 'THERMAL',  x: -3.5 },
    atmos:    { color: 0x15803d, emissive: 0x14532d, label: 'ATMOS',    x: -1.0 },
    nutrient: { color: 0x92400e, emissive: 0x451a03, label: 'NUTRIENTS', x:  1.5 },
    illum:    { color: 0x9a3412, emissive: 0x431407, label: 'ILLUMIN',  x:  4.0 },
  };

  const STATUS_COLORS = {
    nominal: null,
    warning: { color: 0xb45309, emissive: 0x78350f },
    anomaly: { color: 0xb91c1c, emissive: 0x7f1d1d },
  };

  // ── renderer / scene ──────────────────────────────────────────────────────
  const canvas = document.getElementById('selene-habitat-canvas');
  const root   = document.getElementById('selene-habitat-root');
  const W = root.clientWidth  || 800;
  const H = root.clientHeight || 440;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.shadowMap.enabled = true;

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x050508);
  scene.fog = new THREE.Fog(0x050508, 30, 80);

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 200);
  camera.position.set(0, 6, 18);
  camera.lookAt(0, 0, 0);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping    = true;
  controls.dampingFactor    = 0.05;
  controls.autoRotate       = true;
  controls.autoRotateSpeed  = 0.4;
  controls.minDistance      = 8;
  controls.maxDistance      = 40;
  controls.maxPolarAngle    = Math.PI * 0.55;

  // ── lighting ──────────────────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));
  const sun = new THREE.DirectionalLight(0xfff5e0, 1.4);
  sun.position.set(20, 30, 10);
  sun.castShadow = true;
  scene.add(sun);

  // ── stars ─────────────────────────────────────────────────────────────────
  const starGeo  = new THREE.BufferGeometry();
  const starVerts = [];
  for (let i = 0; i < 1800; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = 80 + Math.random() * 20;
    starVerts.push(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
  }
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.18 })));

  // ── lunar surface ─────────────────────────────────────────────────────────
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(120, 120),
    new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.9, metalness: 0.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -2.2;
  ground.receiveShadow = true;
  scene.add(ground);

  // ── Earth ─────────────────────────────────────────────────────────────────
  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(1.6, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0x1a56a0, emissive: 0x0a2040, roughness: 0.6 })
  );
  earth.position.set(-22, 18, -40);
  scene.add(earth);
  // continent tint overlay
  const earthGlow = new THREE.Mesh(
    new THREE.SphereGeometry(1.62, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0x2d6a2d, transparent: true, opacity: 0.35, wireframe: false })
  );
  earth.add(earthGlow);

  // ── habitat shell (cylinder = tunnel) ─────────────────────────────────────
  const shellMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b, transparent: true, opacity: 0.18,
    side: THREE.DoubleSide, roughness: 0.5, metalness: 0.3,
  });
  const shell = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 11, 32, 1, true), shellMat);
  shell.rotation.z = Math.PI / 2;
  scene.add(shell);

  // end caps
  for (const xOff of [-5.5, 5.5]) {
    const cap = new THREE.Mesh(
      new THREE.CircleGeometry(2.2, 32),
      new THREE.MeshStandardMaterial({ color: 0x1e293b, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
    );
    cap.rotation.y = Math.PI / 2;
    cap.position.x = xOff;
    scene.add(cap);
  }

  // structural rings
  for (const xOff of [-4, -1.5, 1, 3.5]) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.2, 0.08, 8, 32),
      new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.3 })
    );
    ring.rotation.y = Math.PI / 2;
    ring.position.x = xOff;
    scene.add(ring);
  }

  // ── subsystem nodes ────────────────────────────────────────────────────────
  const subsystemMeshes = {};
  const subsystemPulse  = {};  // { name: { t, orig, target, active } }

  function makeSubsystem(name, cfg) {
    const group = new THREE.Group();
    group.position.x = cfg.x;

    // control unit box
    const mat = new THREE.MeshStandardMaterial({
      color: cfg.color, emissive: cfg.emissive, emissiveIntensity: 0.4,
      roughness: 0.4, metalness: 0.5,
    });
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.7, 0.7), mat);
    box.castShadow = true;
    group.add(box);

    // pipe
    const pipe = new THREE.Mesh(
      new THREE.CylinderGeometry(0.12, 0.12, 1.6, 10),
      new THREE.MeshStandardMaterial({ color: cfg.color, emissive: cfg.emissive, emissiveIntensity: 0.2, metalness: 0.7 })
    );
    pipe.position.y = -0.9;
    group.add(pipe);

    // sensor markers (3 small spheres)
    for (let i = 0; i < 3; i++) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.1, 8, 8),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      s.position.set(
        (i - 1) * 0.5,
        0.55 + i * 0.15,
        0.4
      );
      group.add(s);
    }

    // point light for ambient glow
    const ptLight = new THREE.PointLight(cfg.color, 0.6, 3.5);
    ptLight.position.set(0, 0.3, 0);
    group.add(ptLight);

    group.userData = { name, mat, ptLight, origColor: cfg.color, origEmissive: cfg.emissive };
    group.userData.clickable = true;

    scene.add(group);
    subsystemMeshes[name] = group;
    subsystemPulse[name]  = { active: false, t: 0 };
  }

  Object.entries(SUBSYSTEMS).forEach(([name, cfg]) => makeSubsystem(name, cfg));

  // ── click detection ────────────────────────────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  renderer.domElement.addEventListener('click', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)   / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(scene.children, true);
    for (const hit of hits) {
      let obj = hit.object;
      while (obj.parent && !obj.userData.clickable) obj = obj.parent;
      if (obj.userData.clickable) {
        window.dispatchEvent(new CustomEvent('selene:subsystem_selected', { detail: obj.userData.name }));
        break;
      }
    }
  });

  // ── status update from agent events ───────────────────────────────────────
  function sensorToSubsystem(sensorId) {
    if (!sensorId) return null;
    if (sensorId.startsWith('tcs'))  return 'thermal';
    if (sensorId.startsWith('ams'))  return 'atmos';
    if (sensorId.startsWith('nds'))  return 'nutrient';
    if (sensorId.startsWith('ics'))  return 'illum';
    return null;
  }

  function setSubsystemStatus(name, status) {
    const group = subsystemMeshes[name];
    if (!group) return;
    const { mat, ptLight, origColor, origEmissive } = group.userData;
    const cfg = STATUS_COLORS[status];
    if (cfg) {
      mat.color.setHex(cfg.color);
      mat.emissive.setHex(cfg.emissive);
      ptLight.color.setHex(cfg.color);
    } else {
      mat.color.setHex(origColor);
      mat.emissive.setHex(origEmissive);
      ptLight.color.setHex(origColor);
    }
    if (status === 'anomaly') {
      subsystemPulse[name] = { active: true, t: 0 };
    }
    // update legend dot
    const legColors = { thermal: '#60a5fa', atmos: '#4ade80', nutrient: '#facc15', illum: '#f97316' };
    const anomalyCol = '#ef4444', warningCol = '#f59e0b';
    const dot = document.getElementById('leg-' + name);
    if (dot) {
      dot.style.background = status === 'anomaly' ? anomalyCol : status === 'warning' ? warningCol : (legColors[name] || '#888');
    }
  }

  window.addEventListener('selene:agent_event', (e) => {
    const ev = e.detail;
    if (!ev) return;
    // AgentRunCompleted carries a diagnosis
    if (ev.type === 'agent_run_completed' && ev.diagnosis) {
      const sensors = ev.diagnosis.affected_sensors || [];
      sensors.forEach(s => {
        const sub = sensorToSubsystem(s);
        if (sub) setSubsystemStatus(sub, 'anomaly');
      });
    }
    // AnomalyEvent
    if (ev.type === 'anomaly' && ev.affected_sensors) {
      ev.affected_sensors.forEach(s => {
        const sub = sensorToSubsystem(s);
        if (sub) setSubsystemStatus(sub, 'warning');
      });
    }
  });

  // ── animation loop ────────────────────────────────────────────────────────
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();

    // pulse anomaly subsystems
    for (const [name, ps] of Object.entries(subsystemPulse)) {
      if (!ps.active) continue;
      ps.t += dt * 2.5;
      const group = subsystemMeshes[name];
      if (group) {
        const intensity = 0.3 + 0.5 * (0.5 + 0.5 * Math.sin(ps.t * Math.PI));
        group.userData.mat.emissiveIntensity = intensity;
      }
    }

    earth.rotation.y += dt * 0.04;
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // ── resize handling ───────────────────────────────────────────────────────
  const ro = new ResizeObserver(() => {
    const w = root.clientWidth, h = root.clientHeight || 440;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  ro.observe(root);
})();
</script>
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
                value=_HABITAT_HTML,
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
