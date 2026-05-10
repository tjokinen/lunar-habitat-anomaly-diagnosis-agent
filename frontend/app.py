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
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr
from gradio.themes import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("selene.frontend")

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

_DATA_PANEL_HTML = """
<div class="selene-panel" id="selene-data-panel">
  <div class="clock-bar">
    <span class="lunar-time" id="lunar-clock">--:--:--</span>
    <span class="speed-indicator" id="speed-indicator" title="Replay speed">—×</span>
    <span class="comm-window" id="comm-window-label">Earth link: 2.6 s RTT</span>
  </div>
  <div class="scenario-timeline" id="scenario-timeline">
    <div class="st-state" id="st-state">No scenario running.</div>
    <div class="st-bar" id="st-bar"><div class="st-fill" id="st-fill"></div></div>
    <div class="st-meta" id="st-meta"></div>
  </div>
  <table class="sensor-table" id="sensor-table">
    <thead>
      <tr>
        <th style="width:52%;text-align:left;font-size:9px;color:#525252;font-weight:400;padding:0 8px 4px;letter-spacing:.06em;">SENSOR</th>
        <th style="width:26%;text-align:right;font-size:9px;color:#525252;font-weight:400;padding:0 4px 4px;letter-spacing:.06em;">VALUE</th>
        <th style="width:22%;text-align:right;font-size:9px;color:#525252;font-weight:400;padding:0 8px 4px;letter-spacing:.06em;">TREND</th>
      </tr>
    </thead>
    <tbody id="sensor-rows">
      <tr><td colspan="3" style="color:#404040;font-size:12px;padding:20px 8px;">
        Waiting for telemetry…
      </td></tr>
    </tbody>
  </table>
</div>
"""

_DATA_PANEL_JS = r"""
() => {
  if (window._seleneDataPanelInit) return;
  window._seleneDataPanelInit = true;

  // ── Sensors to display ────────────────────────────────────────────────────
  const SENSORS = [
    { id: 'tcs/temp-ams_1',       label: 'TCS Temp AMS-1',     unit: '°C'   },
    { id: 'tcs/temp-ams_2',       label: 'TCS Temp AMS-2',     unit: '°C'   },
    { id: 'tcs/pressure-ams',     label: 'TCS Pressure AMS',   unit: 'bar'  },
    { id: 'tcs/rh-ams_1',         label: 'TCS Humidity AMS-1', unit: '%'    },
    { id: 'ams-feg/co2-1',        label: 'CO₂ FEG',            unit: 'ppm'  },
    { id: 'ams-ses/co2-1',        label: 'CO₂ SES',            unit: 'ppm'  },
    { id: 'ams-feg/o2-1',         label: 'O₂ FEG',             unit: '%'    },
    { id: 'nds/level-tank1',      label: 'Nutrient Tank 1',    unit: 'cm'   },
    { id: 'nds/level-tank2',      label: 'Nutrient Tank 2',    unit: 'cm'   },
    { id: 'nds/volume-tank1',     label: 'NDS Volume 1',       unit: 'L'    },
    { id: 'ics/par-1',            label: 'ICS PAR-1',          unit: 'µmol' },
  ];

  // sparkline history: sensorId → [value, ...]  (max 60 points = 5-min cadence × 5 h)
  const HISTORY_LEN = 60;
  const history = {};
  const sensorStatus = {};  // sensorId → 'nominal'|'warning'|'anomaly'
  SENSORS.forEach(s => { history[s.id] = []; sensorStatus[s.id] = 'nominal'; });

  // ── Lunar clock ────────────────────────────────────────────────────────────
  // Drive from telemetry timestamps; fall back to wall clock offset.
  let lastTelemetryTs = null;
  let lastWallMs = Date.now();

  function fmtTime(isoOrDate) {
    const d = typeof isoOrDate === 'string' ? new Date(isoOrDate) : isoOrDate;
    if (isNaN(d)) return '--:--:--';
    return d.toISOString().substring(11, 19);
  }

  // simNow() returns the current simulated time in ms (Unix epoch), derived
  // from the last telemetry timestamp + wall-clock elapsed since then. This
  // lets the comm-window cycle and indicators speed up with --speed N.
  function simNowMs() {
    if (!lastTelemetryTs) return null;
    return new Date(lastTelemetryTs).getTime() + (Date.now() - lastWallMs);
  }

  function tickClock() {
    const clockEl = document.getElementById('lunar-clock');
    if (!clockEl) return;
    const sim = simNowMs();
    if (sim !== null) clockEl.textContent = fmtTime(new Date(sim));
    updateScenarioTimeline();
  }
  setInterval(tickClock, 1000);

  // ── Scenario timeline ──────────────────────────────────────────────────
  // ground_truth = { scenario_id, start_time, end_time, affected_sensors, description }
  // OR null if no scenario is active.
  let scenarioGT = null;
  let scenarioName = null;
  let firstDetectionMs = null;   // wall-clock-aligned sim time of first agent_run_started

  function fmtDur(ms) {
    if (ms == null || isNaN(ms)) return '—';
    const sign = ms < 0 ? '-' : '';
    const s = Math.abs(Math.round(ms / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return sign + h + 'h ' + m + 'm';
    if (m > 0) return sign + m + 'm ' + sec + 's';
    return sign + sec + 's';
  }

  function setTimelineClass(cls) {
    const el = document.getElementById('scenario-timeline');
    if (!el) return;
    el.className = 'scenario-timeline ' + cls;
  }

  function updateScenarioTimeline() {
    const stateEl = document.getElementById('st-state');
    const fillEl  = document.getElementById('st-fill');
    const metaEl  = document.getElementById('st-meta');
    if (!stateEl || !fillEl || !metaEl) return;

    if (!scenarioGT) {
      setTimelineClass('st-idle');
      stateEl.innerHTML = '<span class="st-label">No scenario running.</span>' +
                          '<span class="st-phase">Monitoring nominal telemetry.</span>';
      fillEl.style.width = '0%';
      metaEl.innerHTML = '';
      return;
    }

    const start = new Date(scenarioGT.start_time).getTime();
    const end   = new Date(scenarioGT.end_time).getTime();
    const now   = simNowMs();
    if (now == null || isNaN(start) || isNaN(end) || end <= start) {
      setTimelineClass('st-idle');
      stateEl.innerHTML = '<span class="st-label">' + esc(scenarioName || scenarioGT.scenario_id) + '</span>' +
                          '<span class="st-phase">waiting for telemetry…</span>';
      fillEl.style.width = '0%';
      metaEl.innerHTML = '';
      return;
    }

    const total = end - start;
    let phase, pct, phaseLabel;
    if (now < start) {
      phase = 'st-pre';
      pct = 0;
      phaseLabel = 'Pre-onset · starts in ' + fmtDur(start - now);
    } else if (now < end) {
      phase = 'st-active';
      pct = Math.max(0, Math.min(100, ((now - start) / total) * 100));
      phaseLabel = 'Anomaly active · ' + fmtDur(now - start) + ' / ' + fmtDur(total);
    } else {
      phase = 'st-post';
      pct = 100;
      phaseLabel = 'Anomaly window ended · ' + fmtDur(now - end) + ' ago';
    }

    setTimelineClass(phase);
    stateEl.innerHTML = '<span class="st-label">' + esc(scenarioName || scenarioGT.scenario_id) + '</span>' +
                        '<span class="st-phase">' + esc(phaseLabel) + '</span>';
    fillEl.style.width = pct.toFixed(1) + '%';

    let detectionLine = 'Detection: <span style="color:#737373;">—</span>';
    if (firstDetectionMs != null) {
      const lag = firstDetectionMs - start;
      const lagLabel = lag >= 0
        ? '+' + fmtDur(lag) + ' after onset'
        : fmtDur(lag) + ' before onset';
      detectionLine = 'Detection: <span style="color:#f59e0b;">' +
                      esc(fmtTime(new Date(firstDetectionMs))) + '</span> · ' +
                      esc(lagLabel);
    }

    metaEl.innerHTML =
      '<span>Onset ' + esc(fmtTime(new Date(start))) +
        ' → end ' + esc(fmtTime(new Date(end))) + '</span>' +
      '<span>' + detectionLine + '</span>';
  }

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  window.addEventListener('selene:scenario', (e) => {
    const d = e.detail || {};
    if (d.active && d.ground_truth) {
      scenarioGT = d.ground_truth;
      scenarioName = d.scenario_name || d.scenario_id || null;
      firstDetectionMs = null;   // new scenario → reset detection lag
    } else {
      scenarioGT = null;
      scenarioName = null;
      firstDetectionMs = null;
    }
    updateScenarioTimeline();
  });

  window.addEventListener('selene:first-detection', (e) => {
    const ts = e.detail && e.detail.timestamp;
    if (!ts) return;
    if (firstDetectionMs == null) {
      firstDetectionMs = new Date(ts).getTime();
      updateScenarioTimeline();
    }
  });


  // ── Speed indicator (set by Gradio via window.__seleneSetSpeed) ──────────
  window.__seleneSetSpeed = function(mult) {
    const el = document.getElementById('speed-indicator');
    if (!el) return;
    el.textContent = (mult === null || mult === undefined) ? '∞×' : (mult + '×');
  };

  // ── Sparkline ─────────────────────────────────────────────────────────────
  function drawSparkline(canvas, values) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (values.length < 2) return;

    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * W;
      const y = H - ((v - min) / range) * (H - 2) - 1;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // ── Build / update sensor rows ─────────────────────────────────────────────
  let rowsBuilt = false;

  function buildRows() {
    const tbody = document.getElementById('sensor-rows');
    if (!tbody) return;
    tbody.innerHTML = '';
    SENSORS.forEach(s => {
      const tr = document.createElement('tr');
      tr.id = 'row-' + s.id.replace(/\//g, '-').replace(/_/g, '-');
      tr.innerHTML = `
        <td style="padding:5px 8px;">
          <span class="dot dot-nominal" id="dot-${s.id.replace(/\//g,'-').replace(/_/g,'-')}"></span>
          <span class="sensor-name">${s.label}</span>
        </td>
        <td style="padding:5px 4px;text-align:right;">
          <span class="sensor-value" id="val-${s.id.replace(/\//g,'-').replace(/_/g,'-')}">—</span>
          <span class="sensor-unit">${s.unit}</span>
        </td>
        <td style="padding:3px 8px 3px 4px;text-align:right;vertical-align:middle;">
          <canvas id="spk-${s.id.replace(/\//g,'-').replace(/_/g,'-')}"
                  width="72" height="22"
                  style="display:inline-block;vertical-align:middle;"></canvas>
        </td>`;
      tbody.appendChild(tr);
    });
    rowsBuilt = true;
  }

  function safeId(sensorId) { return sensorId.replace(/\//g,'-').replace(/_/g,'-'); }

  function updateRow(sensorId, value) {
    if (!rowsBuilt) buildRows();
    const sid = safeId(sensorId);
    const valEl = document.getElementById('val-' + sid);
    if (valEl) valEl.textContent = typeof value === 'number' ? value.toFixed(2) : value;

    const hist = history[sensorId];
    if (hist) {
      hist.push(value);
      if (hist.length > HISTORY_LEN) hist.shift();
      const canvas = document.getElementById('spk-' + sid);
      if (canvas) drawSparkline(canvas, hist);
    }
  }

  function setRowStatus(sensorId, status) {
    sensorStatus[sensorId] = status;
    const sid = safeId(sensorId);
    const dot = document.getElementById('dot-' + sid);
    if (!dot) return;
    dot.className = 'dot dot-' + status;
  }

  // ── Telemetry event handler ────────────────────────────────────────────────
  window.addEventListener('selene:telemetry', (e) => {
    const frame = e.detail;
    if (!frame) return;

    // Update clock
    if (frame.timestamp) {
      lastTelemetryTs = frame.timestamp;
      lastWallMs = Date.now();
      const clockEl = document.getElementById('lunar-clock');
      if (clockEl) clockEl.textContent = fmtTime(frame.timestamp);
    }

    if (!rowsBuilt) buildRows();

    // Update each displayed sensor
    const readings = frame.readings || {};
    SENSORS.forEach(s => {
      const r = readings[s.id];
      if (r === undefined || r === null) return;
      // Wire format is the full SensorReading object; pull the numeric value.
      const v = (typeof r === 'object') ? r.value : r;
      if (v !== undefined && v !== null) updateRow(s.id, v);
    });
  });

  // ── Agent event → sensor status ───────────────────────────────────────────
  window.addEventListener('selene:agent_event', (e) => {
    const ev = e.detail; if (!ev) return;
    if (ev.affected_sensors) {
      ev.affected_sensors.forEach(sid => setRowStatus(sid, 'warning'));
    }
    if (ev.type === 'agent_run_completed' && ev.diagnosis) {
      (ev.diagnosis.affected_sensors || []).forEach(sid => setRowStatus(sid, 'anomaly'));
    }
  });

  // Build rows immediately so the panel isn't blank.
  function waitAndBuild() {
    if (document.getElementById('sensor-rows')) { buildRows(); return; }
    setTimeout(waitAndBuild, 200);
  }
  waitAndBuild();
}
"""

_INVESTIGATION_HTML = """
<div class="selene-panel" id="selene-inv-panel" style="min-height:280px;">
  <div class="trace-header">Investigation trace</div>

  <!-- idle banner shown only while there are zero runs -->
  <div id="inv-idle" style="display:flex;align-items:center;gap:8px;color:#404040;font-size:12px;padding:8px 0 12px;">
    <span class="dot dot-nominal" style="animation:pulse-dot 2s infinite;"></span>
    Monitoring nominal — no active investigation.
  </div>

  <!-- Stack of run blocks. Newest investigation sits at the top; older
       runs remain visible below as collapsed cards. -->
  <div id="inv-runs" style="max-height:520px;overflow-y:auto;"></div>
</div>
"""

_INVESTIGATION_JS = r"""
() => {
  if (window._seleneInvInit) return;
  window._seleneInvInit = true;

  // ── per-run state ────────────────────────────────────────────────────────
  // Newest run goes to the front of the list. Each entry holds everything
  // needed to re-render that run's block without reading the DOM.
  // run = { runId, startTs, trigger, status, toolCalls{call_id→tc},
  //         toolOrder[call_id...], hypotheses[], diagnosis, failReason }
  const runs = [];
  function getRun(runId) { return runs.find(r => r.runId === runId); }

  // ── helpers ──────────────────────────────────────────────────────────────
  function el(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function elapsed(startTs, ts) {
    if (!startTs || !ts) return '';
    const dt = Math.round((new Date(ts) - new Date(startTs)) / 1000);
    const m = Math.floor(dt / 60), s = dt % 60;
    return '[+' + (m ? m + ':' + String(s).padStart(2,'0') : s + 's') + ']';
  }

  function shortArgs(args) {
    try {
      const s = JSON.stringify(args);
      if (s === undefined) return '{}';
      return s.length > 80 ? s.slice(0, 77) + '…' : s;
    } catch { return '{}'; }
  }

  function fmtClock(ts) {
    if (!ts) return '';
    try { return new Date(ts).toISOString().substring(11,19); }
    catch { return ''; }
  }

  // ── header HTML for one run ──────────────────────────────────────────────
  function headerHTML(run) {
    const trig = run.trigger || {};
    const sensors = (trig.affected_sensors || []).join(', ') || '—';
    const score   = trig.score != null ? trig.score.toFixed(2) : '';
    const det     = trig.detector_name || '';
    const ts      = fmtClock(run.startTs);

    let title, color;
    if (run.status === 'running') {
      title = `<span class="dot dot-warning" style="animation:pulse-dot 1s infinite;"></span>Investigation in progress`;
      color = '#f59e0b';
    } else if (run.status === 'completed') {
      title = `<span class="dot dot-anomaly"></span>Investigation complete`;
      color = '#ef4444';
    } else if (run.status === 'failed') {
      title = `<span class="dot dot-anomaly"></span>Investigation failed: ${esc(run.failReason || '')}`;
      color = '#ef4444';
    } else {
      title = 'Investigation';
      color = '#a3a3a3';
    }

    return `
      <div style="font-size:12px;color:${color};margin-bottom:4px;">${title}</div>
      <div style="font-size:10px;color:#737373;">
        ${ts ? '<span style="color:#525252;margin-right:8px;">' + esc(ts) + '</span>' : ''}
        Trigger: <span style="color:#d4d4d4;">${esc(sensors)}</span>
        ${score ? ' · score <span style="color:#d4d4d4;">' + score + '</span>' : ''}
        ${det ? ' · detector <span style="color:#d4d4d4;">' + esc(det) + '</span>' : ''}
      </div>`;
  }

  // ── tool log HTML for one run ────────────────────────────────────────────
  function toolLogHTML(run) {
    if (!run.toolOrder || run.toolOrder.length === 0) return '';
    const lines = run.toolOrder.map(cid => {
      const tc = run.toolCalls[cid];
      if (!tc) return '';
      const done = tc.result !== undefined || tc.error;
      const result = tc.error
        ? '<span style="color:#ef4444;">error: ' + esc(tc.error) + '</span>'
        : tc.result
          ? '<span style="color:#ef4444;">' + esc(tc.result) + '</span>'
          : '<span style="color:#525252;">…</span>';
      return `
        <div class="tool-call">
          <span style="color:#475569;margin-right:6px;">${esc(elapsed(run.startTs, tc.startTs))}</span>
          <span class="tc-name">${esc(tc.name || '?')}</span>
          <span class="tc-args">${esc(shortArgs(tc.args))}</span>
          ${done ? '<span class="tc-result"> → ' + result + '</span>' : ''}
        </div>`;
    }).join('');
    return '<div class="tool-log">' + lines + '</div>';
  }

  // ── hypothesis ladder HTML for one run ───────────────────────────────────
  function hypothesesHTML(run) {
    if (!run.hypotheses || run.hypotheses.length === 0) return '';
    return '<div class="trace-header" style="margin-bottom:4px;">Hypotheses</div>' +
      run.hypotheses.map(h => {
        const pct = Math.round(h.confidence * 100);
        return `
          <div class="hypothesis-bar">
            <span class="hb-label" title="${esc(h.id)}">${esc(h.id)}</span>
            <span class="hb-track"><span class="hb-fill" style="width:${pct}%;"></span></span>
            <span class="hb-pct">${pct}%</span>
          </div>`;
      }).join('');
  }

  // ── diagnosis card HTML for one run ──────────────────────────────────────
  function diagnosisHTML(run) {
    const diag = run.diagnosis;
    if (!diag) return '';
    const conf = Math.round((diag.confidence || 0) * 100);
    const fmtList = (arr) => (arr || []).map(x => '<li>' + esc(x) + '</li>').join('');
    const fmtCitations = (arr) => (arr || []).map((c, i) =>
      `<div class="citation">[${i+1}] ${esc(c.title || c.id || '')}` +
      (c.url ? ` — <a href="${esc(c.url)}" target="_blank">${esc(c.url)}</a>` : '') +
      '</div>'
    ).join('');

    return `
      <div class="diagnosis-card">
        <div class="dc-title">${esc(diag.primary_hypothesis || '—')}</div>
        <div class="dc-conf">
          Confidence: ${conf}%
          <span style="display:inline-block;width:80px;height:5px;background:#1f1f1f;border-radius:2px;margin-left:8px;vertical-align:middle;">
            <span style="display:block;height:5px;width:${conf}%;background:${conf>=70?'#ef4444':conf>=40?'#f59e0b':'#737373'};border-radius:2px;"></span>
          </span>
        </div>
        ${diag.matched_failure_modes && diag.matched_failure_modes.length ? `
          <div class="dc-section">Matched failure modes</div>
          <ul>${fmtList(diag.matched_failure_modes)}</ul>` : ''}
        ${diag.supporting_evidence && diag.supporting_evidence.length ? `
          <div class="dc-section">Supporting evidence</div>
          <ul>${fmtList(diag.supporting_evidence)}</ul>` : ''}
        ${diag.recommended_actions && diag.recommended_actions.length ? `
          <div class="dc-section">Recommended actions</div>
          <ul>${fmtList(diag.recommended_actions)}</ul>` : ''}
        ${diag.citations && diag.citations.length ? `
          <div class="dc-section">References</div>
          ${fmtCitations(diag.citations)}` : ''}
      </div>`;
  }

  // ── render the full stack of run blocks ──────────────────────────────────
  function render() {
    const root = el('inv-runs');
    if (!root) return;
    const idle = el('inv-idle');
    if (idle) idle.style.display = runs.length === 0 ? '' : 'none';

    root.innerHTML = runs.map((run, idx) => {
      const cls = run.status === 'running'
        ? 'inv-run-block inv-run-active'
        : 'inv-run-block inv-run-past';
      return `
        <div class="${cls}" data-run="${esc(run.runId)}">
          ${headerHTML(run)}
          ${toolLogHTML(run)}
          ${hypothesesHTML(run)}
          ${diagnosisHTML(run)}
        </div>`;
    }).join('');
  }

  // ── event dispatch ───────────────────────────────────────────────────────
  window.addEventListener('selene:agent_event', (e) => {
    const ev = e.detail;
    if (!ev || !ev.type) return;

    if (ev.type === 'agent_run_started') {
      // Mark any still-running run as failed (lost) before pushing new one.
      runs.forEach(r => { if (r.status === 'running') r.status = 'failed'; });
      runs.unshift({
        runId: ev.run_id,
        startTs: ev.timestamp,
        trigger: ev.trigger || {},
        status: 'running',
        toolCalls: {},
        toolOrder: [],
        hypotheses: [],
        diagnosis: null,
        failReason: null,
      });
      // Cap retained history.
      if (runs.length > 12) runs.length = 12;
      // Notify the data panel so the scenario timeline can show
      // detection lag (it dedups internally).
      window.dispatchEvent(new CustomEvent('selene:first-detection',
        { detail: { timestamp: ev.timestamp } }));
      render();
    }

    if (ev.type === 'tool_call_started') {
      const run = getRun(ev.run_id);
      if (!run) return;
      if (!(ev.call_id in run.toolCalls)) run.toolOrder.push(ev.call_id);
      run.toolCalls[ev.call_id] = {
        name: ev.tool_name, args: ev.arguments, startTs: ev.timestamp,
      };
      render();
    }

    if (ev.type === 'tool_call_completed') {
      const run = getRun(ev.run_id);
      if (!run) return;
      const tc = run.toolCalls[ev.call_id] || {};
      // Fallback: if started was missed (e.g. WS reconnect mid-run),
      // completed still carries name/args.
      if (tc.name === undefined)    tc.name    = ev.tool_name;
      if (tc.args === undefined)    tc.args    = ev.arguments;
      if (tc.startTs === undefined) tc.startTs = ev.timestamp;
      tc.result = ev.result_summary;
      tc.error  = ev.error || null;
      if (!(ev.call_id in run.toolCalls)) run.toolOrder.push(ev.call_id);
      run.toolCalls[ev.call_id] = tc;
      render();
    }

    if (ev.type === 'hypothesis_ladder_updated') {
      const run = getRun(ev.run_id);
      if (!run) return;
      run.hypotheses = (ev.ranked || []).map(([id, conf]) => ({ id, confidence: conf }));
      render();
    }

    if (ev.type === 'agent_run_completed') {
      const run = getRun(ev.run_id);
      if (!run) return;
      run.status = 'completed';
      run.diagnosis = ev.diagnosis || null;
      render();
    }

    if (ev.type === 'agent_run_failed') {
      const run = getRun(ev.run_id);
      if (!run) return;
      run.status = 'failed';
      run.failReason = ev.reason;
      render();
    }
  });
}
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
    msg_count = 0
    while True:
        try:
            logger.info("WS pump connecting: %s", url)
            async with websockets.connect(url) as ws:
                logger.info("WS pump connected: %s", url)
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    msg_count += 1
                    if msg_count <= 3 or msg_count % 50 == 0:
                        logger.info("WS pump %s msg #%d", endpoint, msg_count)
                    await queue.put({"type": event_type, payload_key: data})
        except Exception as exc:
            logger.warning("WS pump %s error: %s", endpoint, exc)
        await asyncio.sleep(_RECONNECT_DELAY_SECS)


async def _event_stream() -> AsyncGenerator[str, None]:
    """Merge telemetry + agent_event WS streams.

    Yields JSON strings to a hidden gr.Textbox whose value the browser-side
    MutationObserver picks up.  We append a monotonic suffix in a separate
    field so two consecutive identical payloads still mutate the textarea.
    """
    logger.info("event_stream started; BACKEND_URL=%s BACKEND_WS_URL=%s",
                BACKEND_URL, BACKEND_WS_URL)
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump("/telemetry",    "telemetry",    "frame", queue)),
        asyncio.create_task(_pump("/agent_events", "agent_event",  "event", queue)),
    ]
    yielded = 0
    try:
        while True:
            item = await queue.get()
            yielded += 1
            item["_seq"] = yielded
            if yielded <= 3 or yielded % 50 == 0:
                logger.info("event_stream yield #%d type=%s", yielded, item.get("type"))
            yield json.dumps(item)
    finally:
        for t in tasks:
            t.cancel()


# Browser-side MutationObserver that converts event_state mutations into
# custom DOM events dispatched on window.
_OBSERVER_JS = """
() => {
  let mutCount = 0, dispatchCount = 0, parseFails = 0;
  let lastSeq = -1;
  window.__seleneDebug = function() {
    console.log('[selene] mutations=' + mutCount +
                ' dispatched=' + dispatchCount +
                ' parseFails=' + parseFails +
                ' lastSeq=' + lastSeq);
  };

  function dispatch(raw) {
    try {
      const data = JSON.parse(raw || '{}');
      if (!data || !data.type) return;
      if (data._seq !== undefined) lastSeq = data._seq;
      if (data.type === 'telemetry') {
        window.dispatchEvent(new CustomEvent('selene:telemetry', { detail: data.frame }));
        dispatchCount++;
      } else if (data.type === 'agent_event') {
        window.dispatchEvent(new CustomEvent('selene:agent_event', { detail: data.event }));
        dispatchCount++;
      }
    } catch(_) { parseFails++; }
  }

  function attachSeleneObserver() {
    const root = document.querySelector('#event-state');
    if (!root) { setTimeout(attachSeleneObserver, 200); return; }
    const ta = root.querySelector('textarea') || root.querySelector('input');
    if (!ta) { setTimeout(attachSeleneObserver, 200); return; }
    console.log('[selene] observer attached to', ta.tagName, '#event-state');

    // 1. Poll the textarea value — most robust across Gradio versions, since
    //    Gradio sets .value programmatically (no native input event fires).
    let lastVal = '';
    setInterval(() => {
      if (ta.value !== lastVal) {
        lastVal = ta.value;
        mutCount++;
        dispatch(ta.value);
      }
    }, 100);

    // 2. Belt-and-suspenders: also watch for DOM mutations (in case Gradio
    //    re-renders the textarea node).
    const obs = new MutationObserver(() => {
      if (ta.value !== lastVal) {
        lastVal = ta.value;
        mutCount++;
        dispatch(ta.value);
      }
    });
    obs.observe(root, { childList: true, subtree: true, characterData: true, attributes: true });
  }
  attachSeleneObserver();

  // ── Scenario-state observer (separate textbox, fires `selene:scenario`) ──
  function dispatchScenario(raw) {
    if (!raw) return;
    try {
      const data = JSON.parse(raw);
      window.dispatchEvent(new CustomEvent('selene:scenario', { detail: data }));
    } catch(_) {}
  }
  function attachScenarioObserver() {
    const root = document.querySelector('#scenario-state');
    if (!root) { setTimeout(attachScenarioObserver, 200); return; }
    const ta = root.querySelector('textarea') || root.querySelector('input');
    if (!ta) { setTimeout(attachScenarioObserver, 200); return; }
    console.log('[selene] scenario observer attached');
    let lastVal = '';
    setInterval(() => {
      if (ta.value !== lastVal) { lastVal = ta.value; dispatchScenario(ta.value); }
    }, 200);
    new MutationObserver(() => {
      if (ta.value !== lastVal) { lastVal = ta.value; dispatchScenario(ta.value); }
    }).observe(root, { childList: true, subtree: true, characterData: true, attributes: true });
  }
  attachScenarioObserver();
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

            # Scenario controls
            with gr.Row(elem_classes=["controls-row"]):
                scenario_dropdown = gr.Dropdown(
                    choices=[],
                    label="Scenario",
                    scale=3,
                    interactive=True,
                )
                start_btn = gr.Button("▶ Start", variant="primary", scale=1)
                reset_btn = gr.Button("↺ Reset", variant="secondary", scale=1)
                speed_dropdown = gr.Dropdown(
                    choices=[
                        ("60×",   60.0),
                        ("300×",  300.0),
                        ("600×",  600.0),
                        ("1800×", 1800.0),
                        ("3600×", 3600.0),
                    ],
                    value=60.0,
                    label="Speed",
                    scale=1,
                    interactive=True,
                )

            scenario_desc = gr.Markdown(
                value="",
                elem_id="scenario-desc",
            )
            status_label = gr.Markdown("**Status:** Idle")

        # Right column — live data panel + investigation trace
        with gr.Column(scale=2):
            data_panel = gr.HTML(
                value=_DATA_PANEL_HTML,
                elem_id="data-panel",
            )
            investigation_panel = gr.HTML(
                value=_INVESTIGATION_HTML,
                elem_id="investigation-panel",
            )

    # Hidden Textbox — the WebSocket pump writes a JSON string here;
    # browser-side MutationObserver watches the textarea for changes and
    # dispatches custom events.  We use Textbox rather than gr.JSON because
    # gr.JSON's tree-viewer component does not produce reliable DOM
    # mutations when streamed via async-generator yields, and `visible=False`
    # may strip the element from the DOM entirely in newer Gradio versions.
    event_state = gr.Textbox(
        value="",
        elem_id="event-state",
        elem_classes=["selene-hidden"],
        show_label=False,
        lines=1,
        max_lines=1,
        interactive=False,
    )

    # Hidden Textbox carrying the active scenario + ground-truth window as
    # JSON.  Written by _start_scenario / _reset_scenario / _load_active_scenario
    # and observed by the scenario-timeline JS (selene:scenario events).
    scenario_state = gr.Textbox(
        value="",
        elem_id="scenario-state",
        elem_classes=["selene-hidden"],
        show_label=False,
        lines=1,
        max_lines=1,
        interactive=False,
    )

    # scenario_id → {name, description} — filled on load, read by desc handler
    _scenario_meta: dict[str, dict] = {}

    # ── Populate scenario dropdown on load ─────────────────────────────────
    async def _load_scenarios() -> tuple[gr.Dropdown, str]:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{BACKEND_URL}/scenarios")
                resp.raise_for_status()
                scenarios = resp.json()
            for s in scenarios:
                _scenario_meta[s["scenario_id"]] = {
                    "name":        s.get("name", s["scenario_id"]),
                    "description": s.get("description", ""),
                }
            # (display_name, scenario_id) tuples so dropdown shows friendly names
            choices = [(m["name"], sid) for sid, m in _scenario_meta.items()]
            first_id = scenarios[0]["scenario_id"] if scenarios else None
            first_desc = _fmt_desc(first_id) if first_id else ""
            return gr.Dropdown(choices=choices, value=first_id), first_desc
        except Exception:
            return gr.Dropdown(choices=[], value=None), ""

    def _fmt_desc(scenario_id: str | None) -> str:
        if not scenario_id or scenario_id not in _scenario_meta:
            return ""
        m = _scenario_meta[scenario_id]
        desc = m["description"]
        return f"*{desc}*" if desc else ""

    demo.load(fn=_load_scenarios, inputs=[], outputs=[scenario_dropdown, scenario_desc])

    # ── Attach MutationObserver + Three.js habitat on page load ───────────
    demo.load(fn=None, js=_OBSERVER_JS)
    demo.load(fn=None, js=_HABITAT_JS)
    demo.load(fn=None, js=_DATA_PANEL_JS)
    demo.load(fn=None, js=_INVESTIGATION_JS)

    # ── WebSocket event pump — streams backend events into event_state ─────
    demo.load(fn=_event_stream, outputs=[event_state])

    # ── Scenario description on dropdown change ────────────────────────────
    scenario_dropdown.change(
        fn=_fmt_desc,
        inputs=[scenario_dropdown],
        outputs=[scenario_desc],
    )

    # ── Scenario start / reset ─────────────────────────────────────────────
    def _scenario_payload_json(payload: dict, scenario_id: str | None) -> str:
        """Augment a /scenario/* JSON response with the friendly scenario name
        and dump to a JSON string for the hidden #scenario-state textbox."""
        if scenario_id and scenario_id in _scenario_meta:
            payload = {**payload, "scenario_name": _scenario_meta[scenario_id]["name"]}
        return json.dumps(payload)

    async def _start_scenario(scenario_id: str | None):
        if not scenario_id:
            return (
                "**Status:** No scenario selected",
                json.dumps({"active": False, "scenario_id": None, "ground_truth": None}),
                gr.Button(interactive=True),
            )
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{BACKEND_URL}/scenario/start",
                    json={"scenario_id": scenario_id},
                )
                resp.raise_for_status()
                data = resp.json()
            name = _scenario_meta.get(scenario_id, {}).get("name", scenario_id)
            return (
                f"**Status:** Running — {name}",
                _scenario_payload_json(data, scenario_id),
                gr.Button(interactive=False),
            )
        except Exception as exc:
            return f"**Status:** Error — {exc}", "", gr.Button(interactive=True)

    async def _reset_scenario():
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{BACKEND_URL}/scenario/reset")
                resp.raise_for_status()
                data = resp.json()
            return "**Status:** Nominal (reset)", _scenario_payload_json(data, None), gr.Button(interactive=True)
        except Exception as exc:
            return f"**Status:** Error — {exc}", "", gr.Button(interactive=True)

    async def _load_active_scenario() -> str:
        """Fetched on page load so a refresh restores the scenario timeline."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{BACKEND_URL}/scenario/active")
                resp.raise_for_status()
                data = resp.json()
            sid = data.get("scenario_id")
            return _scenario_payload_json(data, sid)
        except Exception:
            return ""

    start_btn.click(
        fn=_start_scenario,
        inputs=[scenario_dropdown],
        outputs=[status_label, scenario_state, start_btn],
    )
    reset_btn.click(
        fn=_reset_scenario,
        inputs=[],
        outputs=[status_label, scenario_state, start_btn],
    )
    demo.load(fn=_load_active_scenario, inputs=[], outputs=[scenario_state])

    # ── Replay speed: load current value + push UI changes to backend ──────
    async def _load_speed() -> float:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{BACKEND_URL}/replay/speed")
                resp.raise_for_status()
                return resp.json().get("multiplier") or 60.0
        except Exception:
            return 60.0

    async def _set_speed(multiplier: float | None) -> None:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.post(
                    f"{BACKEND_URL}/replay/speed",
                    json={"multiplier": multiplier},
                )
        except Exception as exc:
            logger.warning("set_speed failed: %s", exc)

    demo.load(fn=_load_speed, inputs=[], outputs=[speed_dropdown])

    # Mirror the dropdown into the on-screen indicator via window.__seleneSetSpeed.
    speed_dropdown.change(
        fn=_set_speed,
        inputs=[speed_dropdown],
        outputs=[],
        js="(v) => { if (window.__seleneSetSpeed) window.__seleneSetSpeed(v); return v; }",
    )
    # Also push the loaded value into the indicator on first load.
    demo.load(
        fn=None,
        inputs=[speed_dropdown],
        outputs=[],
        js="(v) => { if (window.__seleneSetSpeed) window.__seleneSetSpeed(v); return v; }",
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
