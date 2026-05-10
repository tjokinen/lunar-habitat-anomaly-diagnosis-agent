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
  <canvas id="selene-habitat-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;display:block;"></canvas>
  <div id="selene-habitat-status" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-family:monospace;font-size:11px;color:#404040;pointer-events:none;">
    initializing scene…
  </div>
  <div id="selene-legend" style="position:absolute;bottom:10px;left:12px;font-family:monospace;font-size:10px;color:#d4d4d4;display:flex;gap:12px;pointer-events:none;display:none;">
    <span><span id="leg-thermal"  style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#60a5fa;margin-right:4px;vertical-align:middle;"></span>THERMAL</span>
    <span><span id="leg-atmos"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;margin-right:4px;vertical-align:middle;"></span>ATMOS</span>
    <span><span id="leg-nutrient" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#facc15;margin-right:4px;vertical-align:middle;"></span>NUTRIENTS</span>
    <span><span id="leg-illum"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f97316;margin-right:4px;vertical-align:middle;"></span>ILLUMIN</span>
  </div>
  <div style="position:absolute;top:10px;right:12px;font-family:monospace;font-size:10px;color:#475569;pointer-events:none;">
    Earth — comm delay: 2.6 s
  </div>
</div>
"""

# Three.js scene — injected via demo.load(js=) because gr.HTML strips <script> tags.
_HABITAT_JS = """
() => {
  // Allow re-init if previous attempt failed; guard against true double-init.
  if (window._seleneHabitatRunning) return;
  window._seleneHabitatRunning = true;

  function setStatus(msg) {
    const el = document.getElementById('selene-habitat-status');
    if (el) el.textContent = msg;
  }

  function loadScript(src, onload, onerror) {
    if (document.querySelector('script[src="' + src + '"]')) { onload(); return; }
    const s = document.createElement('script');
    s.src = src;
    s.onload = onload;
    s.onerror = onerror || (() => setStatus('failed to load: ' + src));
    document.head.appendChild(s);
  }

  // Wait for canvas AND non-zero dimensions (Gradio may render after JS fires).
  function waitForCanvas(cb, elapsed) {
    elapsed = elapsed || 0;
    const root   = document.getElementById('selene-habitat-root');
    const canvas = document.getElementById('selene-habitat-canvas');
    if (root && canvas && root.getBoundingClientRect().width > 10) {
      cb(canvas, root);
      return;
    }
    if (elapsed > 12000) { setStatus('canvas not found'); return; }
    setTimeout(() => waitForCanvas(cb, elapsed + 200), 200);
  }

  setStatus('loading three.js…');

  function initScene() {
    setStatus('waiting for canvas…');
    waitForCanvas((canvas, root) => {
      setStatus('building scene…');
      const rect = root.getBoundingClientRect();
      const W = Math.round(rect.width)  || 800;
      const H = Math.round(rect.height) || 440;

      const SUBSYSTEMS = {
        thermal:  { color: 0x1d4ed8, emissive: 0x1e3a5f, x: -3.5 },
        atmos:    { color: 0x15803d, emissive: 0x14532d, x: -1.0 },
        nutrient: { color: 0x92400e, emissive: 0x451a03, x:  1.5 },
        illum:    { color: 0x9a3412, emissive: 0x431407, x:  4.0 },
      };
      const STATUS_COLORS = {
        warning: { color: 0xb45309, emissive: 0x78350f },
        anomaly: { color: 0xb91c1c, emissive: 0x7f1d1d },
      };

      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.shadowMap.enabled = true;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050508);
      scene.fog = new THREE.Fog(0x050508, 30, 80);

      const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 200);
      camera.position.set(0, 6, 18);
      camera.lookAt(0, 0, 0);

      // Minimal orbit controls (Three.js r160 dropped examples/js/).
      const controls = (() => {
        let down = false, lastX = 0, lastY = 0;
        const sph = {
          theta: Math.atan2(camera.position.x, camera.position.z),
          phi:   Math.acos(Math.max(-1, Math.min(1, camera.position.y / camera.position.length()))),
          r:     camera.position.length(),
        };
        const MIN_R = 8, MAX_R = 40, MAX_PHI = Math.PI * 0.55;
        function apply() {
          sph.phi = Math.max(0.1, Math.min(MAX_PHI, sph.phi));
          sph.r   = Math.max(MIN_R, Math.min(MAX_R, sph.r));
          camera.position.set(
            sph.r * Math.sin(sph.phi) * Math.sin(sph.theta),
            sph.r * Math.cos(sph.phi),
            sph.r * Math.sin(sph.phi) * Math.cos(sph.theta)
          );
          camera.lookAt(0, 0, 0);
        }
        const el = renderer.domElement;
        el.addEventListener('pointerdown', e => { down = true; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture(e.pointerId); });
        el.addEventListener('pointerup',   () => { down = false; });
        el.addEventListener('pointermove', e => {
          if (!down) return;
          sph.theta -= (e.clientX - lastX) * 0.008;
          sph.phi   -= (e.clientY - lastY) * 0.008;
          lastX = e.clientX; lastY = e.clientY;
          apply();
        });
        el.addEventListener('wheel', e => { sph.r += e.deltaY * 0.04; apply(); e.preventDefault(); }, { passive: false });
        apply();
        return { update() { sph.theta += 0.003; apply(); } };
      })();

      scene.add(new THREE.AmbientLight(0xffffff, 0.15));
      const sun = new THREE.DirectionalLight(0xfff5e0, 1.4);
      sun.position.set(20, 30, 10); sun.castShadow = true; scene.add(sun);

      // stars
      const starVerts = [];
      for (let i = 0; i < 1800; i++) {
        const t = Math.random() * Math.PI * 2, p = Math.acos(2 * Math.random() - 1), r = 80 + Math.random() * 20;
        starVerts.push(r*Math.sin(p)*Math.cos(t), r*Math.sin(p)*Math.sin(t), r*Math.cos(p));
      }
      const sg = new THREE.BufferGeometry();
      sg.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
      scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 0.18 })));

      // lunar ground
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(120,120),
        new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.9 }));
      ground.rotation.x = -Math.PI/2; ground.position.y = -2.2; ground.receiveShadow = true;
      scene.add(ground);

      // Earth
      const earth = new THREE.Mesh(new THREE.SphereGeometry(1.6,32,32),
        new THREE.MeshStandardMaterial({ color: 0x1a56a0, emissive: 0x0a2040, roughness: 0.6 }));
      earth.position.set(-22, 18, -40); scene.add(earth);
      earth.add(new THREE.Mesh(new THREE.SphereGeometry(1.62,32,32),
        new THREE.MeshBasicMaterial({ color: 0x2d6a2d, transparent: true, opacity: 0.35 })));

      // habitat shell
      const shell = new THREE.Mesh(new THREE.CylinderGeometry(2.2,2.2,11,32,1,true),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, transparent: true, opacity: 0.18, side: THREE.DoubleSide, roughness:0.5, metalness:0.3 }));
      shell.rotation.z = Math.PI/2; scene.add(shell);
      for (const xOff of [-5.5, 5.5]) {
        const cap = new THREE.Mesh(new THREE.CircleGeometry(2.2,32),
          new THREE.MeshStandardMaterial({ color: 0x1e293b, transparent:true, opacity:0.35, side:THREE.DoubleSide }));
        cap.rotation.y = Math.PI/2; cap.position.x = xOff; scene.add(cap);
      }
      for (const xOff of [-4,-1.5,1,3.5]) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2,0.08,8,32),
          new THREE.MeshStandardMaterial({ color: 0x334155, metalness:0.7, roughness:0.3 }));
        ring.rotation.y = Math.PI/2; ring.position.x = xOff; scene.add(ring);
      }

      // subsystem nodes
      const subsystemMeshes = {}, subsystemPulse = {};
      for (const [name, cfg] of Object.entries(SUBSYSTEMS)) {
        const g = new THREE.Group(); g.position.x = cfg.x;
        const mat = new THREE.MeshStandardMaterial({ color:cfg.color, emissive:cfg.emissive, emissiveIntensity:0.4, roughness:0.4, metalness:0.5 });
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.7,0.7,0.7), mat); box.castShadow=true; g.add(box);
        const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.12,1.6,10),
          new THREE.MeshStandardMaterial({ color:cfg.color, emissive:cfg.emissive, emissiveIntensity:0.2, metalness:0.7 }));
        pipe.position.y = -0.9; g.add(pipe);
        for (let i=0;i<3;i++) {
          const s = new THREE.Mesh(new THREE.SphereGeometry(0.1,8,8), new THREE.MeshBasicMaterial({color:0xffffff}));
          s.position.set((i-1)*0.5, 0.55+i*0.15, 0.4); g.add(s);
        }
        const ptLight = new THREE.PointLight(cfg.color, 0.6, 3.5); ptLight.position.set(0,0.3,0); g.add(ptLight);
        g.userData = { name, mat, ptLight, origColor:cfg.color, origEmissive:cfg.emissive, clickable:true };
        scene.add(g);
        subsystemMeshes[name] = g;
        subsystemPulse[name]  = { active:false, t:0 };
      }

      // click handler
      const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
      renderer.domElement.addEventListener('click', (e) => {
        const r = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX-r.left)/r.width)*2-1;
        mouse.y = -((e.clientY-r.top)/r.height)*2+1;
        ray.setFromCamera(mouse, camera);
        for (const hit of ray.intersectObjects(scene.children, true)) {
          let o = hit.object;
          while (o.parent && !o.userData.clickable) o = o.parent;
          if (o.userData.clickable) {
            window.dispatchEvent(new CustomEvent('selene:subsystem_selected', { detail: o.userData.name }));
            break;
          }
        }
      });

      // status updates
      const LEG_COLORS = { thermal:'#60a5fa', atmos:'#4ade80', nutrient:'#facc15', illum:'#f97316' };
      function setSubsystemStatus(name, status) {
        const g = subsystemMeshes[name]; if (!g) return;
        const { mat, ptLight, origColor, origEmissive } = g.userData;
        const cfg = STATUS_COLORS[status];
        if (cfg) { mat.color.setHex(cfg.color); mat.emissive.setHex(cfg.emissive); ptLight.color.setHex(cfg.color); }
        else { mat.color.setHex(origColor); mat.emissive.setHex(origEmissive); ptLight.color.setHex(origColor); }
        if (status === 'anomaly') subsystemPulse[name] = { active:true, t:0 };
        const dot = document.getElementById('leg-'+name);
        if (dot) dot.style.background = status==='anomaly'?'#ef4444':status==='warning'?'#f59e0b':(LEG_COLORS[name]||'#888');
      }
      function sensorToSub(id) {
        if (!id) return null;
        if (id.startsWith('tcs')) return 'thermal';
        if (id.startsWith('ams')) return 'atmos';
        if (id.startsWith('nds')) return 'nutrient';
        if (id.startsWith('ics')) return 'illum';
        return null;
      }
      window.addEventListener('selene:agent_event', (e) => {
        const ev = e.detail; if (!ev) return;
        if (ev.type === 'agent_run_completed' && ev.diagnosis) {
          (ev.diagnosis.affected_sensors||[]).forEach(s => { const sub=sensorToSub(s); if(sub) setSubsystemStatus(sub,'anomaly'); });
        }
        if (ev.type === 'anomaly' && ev.affected_sensors) {
          ev.affected_sensors.forEach(s => { const sub=sensorToSub(s); if(sub) setSubsystemStatus(sub,'warning'); });
        }
      });

      // hide status, show legend once scene is ready
      const statusEl = document.getElementById('selene-habitat-status');
      if (statusEl) statusEl.style.display = 'none';
      const legendEl = document.getElementById('selene-legend');
      if (legendEl) legendEl.style.display = 'flex';

      // animation loop
      const clock = new THREE.Clock();
      (function animate() {
        requestAnimationFrame(animate);
        const dt = clock.getDelta();
        for (const [name, ps] of Object.entries(subsystemPulse)) {
          if (!ps.active) continue;
          ps.t += dt*2.5;
          const g = subsystemMeshes[name];
          if (g) g.userData.mat.emissiveIntensity = 0.3 + 0.5*(0.5+0.5*Math.sin(ps.t*Math.PI));
        }
        earth.rotation.y += dt*0.04;
        controls.update();
        renderer.render(scene, camera);
      })();

      // resize
      new ResizeObserver(() => {
        const r2 = root.getBoundingClientRect();
        const w = Math.round(r2.width) || 800, h = Math.round(r2.height) || 440;
        renderer.setSize(w, h); camera.aspect = w/h; camera.updateProjectionMatrix();
      }).observe(root);
    });
  }

  loadScript(
    'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js',
    initScene
  );
}
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

    # ── Attach MutationObserver + Three.js habitat on page load ───────────
    demo.load(fn=None, js=_OBSERVER_JS)
    demo.load(fn=None, js=_HABITAT_JS)

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
