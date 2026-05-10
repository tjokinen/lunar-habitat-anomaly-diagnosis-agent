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

_DATA_PANEL_HTML = """
<div class="selene-panel" id="selene-data-panel">
  <div class="clock-bar">
    <span class="lunar-time" id="lunar-clock">--:--:--</span>
    <span class="comm-window" id="comm-window-label">Next pass: —</span>
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

  function tickClock() {
    const clockEl = document.getElementById('lunar-clock');
    if (!clockEl) return;
    if (lastTelemetryTs) {
      const elapsed = Date.now() - lastWallMs;
      const d = new Date(new Date(lastTelemetryTs).getTime() + elapsed);
      clockEl.textContent = fmtTime(d);
    }
  }
  setInterval(tickClock, 1000);

  // ── Comm window (simple: cycles every 2h, 10-min window) ──────────────────
  function updateCommWindow() {
    const el = document.getElementById('comm-window-label');
    if (!el) return;
    const now = Date.now();
    const CYCLE_MS = 2 * 3600 * 1000, WINDOW_MS = 10 * 60 * 1000;
    const phase = now % CYCLE_MS;
    if (phase < WINDOW_MS) {
      const rem = Math.ceil((WINDOW_MS - phase) / 60000);
      el.textContent = 'Comm window open — ' + rem + 'm left';
      el.style.color = '#22c55e';
    } else {
      const next = Math.ceil((CYCLE_MS - phase) / 60000);
      const h = Math.floor(next / 60), m = next % 60;
      el.textContent = 'Next pass: ' + (h ? h + 'h ' : '') + m + 'm';
      el.style.color = '';
    }
  }
  setInterval(updateCommWindow, 15000);
  updateCommWindow();

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
      if (r !== undefined && r !== null) updateRow(s.id, r);
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

  <!-- idle state -->
  <div id="inv-idle">
    <div id="inv-idle-pulse" style="display:flex;align-items:center;gap:8px;color:#404040;font-size:12px;padding:8px 0 12px;">
      <span class="dot dot-nominal" style="animation:pulse-dot 2s infinite;"></span>
      Monitoring nominal — no active investigation.
    </div>
    <div id="inv-detector-log" style="font-size:10px;color:#525252;font-family:monospace;max-height:120px;overflow-y:auto;"></div>
  </div>

  <!-- active investigation state (hidden until agent fires) -->
  <div id="inv-active" style="display:none;">
    <div id="inv-header" style="margin-bottom:8px;"></div>
    <div id="inv-tool-log" style="max-height:160px;overflow-y:auto;margin-bottom:8px;"></div>
    <div id="inv-hypotheses" style="margin-bottom:8px;"></div>
    <div id="inv-diagnosis" style="display:none;"></div>
  </div>
</div>
"""

_INVESTIGATION_JS = r"""
() => {
  if (window._seleneInvInit) return;
  window._seleneInvInit = true;

  // per-run state
  let activeRunId   = null;
  let runStartTs    = null;
  const toolCalls   = {};   // call_id → { name, args, startTs, result, error }
  const hypotheses  = [];   // [{ id, confidence }]
  const detectorLog = [];   // last N anomaly lines

  // ── helpers ────────────────────────────────────────────────────────────────
  function el(id) { return document.getElementById(id); }

  function elapsed(ts) {
    if (!runStartTs || !ts) return '';
    const dt = Math.round((new Date(ts) - new Date(runStartTs)) / 1000);
    const m = Math.floor(dt / 60), s = dt % 60;
    return '[+' + (m ? m + ':' + String(s).padStart(2,'0') : s + 's') + ']';
  }

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function shortArgs(args) {
    try {
      const s = JSON.stringify(args);
      return s.length > 80 ? s.slice(0, 77) + '…' : s;
    } catch { return '{}'; }
  }

  // ── detector log (idle state) ─────────────────────────────────────────────
  function pushDetectorLog(line) {
    detectorLog.push(line);
    if (detectorLog.length > 8) detectorLog.shift();
    const logEl = el('inv-detector-log');
    if (!logEl) return;
    logEl.innerHTML = detectorLog.map(l =>
      '<div style="padding:1px 0;border-bottom:1px solid #1a1a1a;">' + esc(l) + '</div>'
    ).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ── show/hide states ──────────────────────────────────────────────────────
  function showIdle() {
    const a = el('inv-active'), i = el('inv-idle');
    if (a) a.style.display = 'none';
    if (i) i.style.display = '';
  }
  function showActive() {
    const a = el('inv-active'), i = el('inv-idle');
    if (a) a.style.display = '';
    if (i) i.style.display = 'none';
  }

  // ── render tool log ───────────────────────────────────────────────────────
  function renderToolLog() {
    const logEl = el('inv-tool-log');
    if (!logEl) return;
    logEl.innerHTML = Object.values(toolCalls).map(tc => {
      const done    = tc.result !== undefined || tc.error;
      const result  = tc.error
        ? '<span style="color:#ef4444;">error: ' + esc(tc.error) + '</span>'
        : tc.result
          ? '<span style="color:#4ade80;">' + esc(tc.result) + '</span>'
          : '<span style="color:#525252;">…</span>';
      return `
        <div class="tool-call">
          <span style="color:#475569;margin-right:6px;">${esc(elapsed(tc.startTs))}</span>
          <span class="tc-name">${esc(tc.name)}</span>
          <span class="tc-args">${esc(shortArgs(tc.args))}</span>
          ${done ? '<span class="tc-result"> → ' + result + '</span>' : ''}
        </div>`;
    }).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ── render hypothesis ladder ──────────────────────────────────────────────
  function renderHypotheses() {
    const el2 = el('inv-hypotheses');
    if (!el2 || hypotheses.length === 0) return;
    el2.innerHTML = '<div class="trace-header" style="margin-bottom:4px;">Hypotheses</div>' +
      hypotheses.map(h => {
        const pct = Math.round(h.confidence * 100);
        return `
          <div class="hypothesis-bar">
            <span class="hb-label" title="${esc(h.id)}">${esc(h.id)}</span>
            <span class="hb-track"><span class="hb-fill" style="width:${pct}%;"></span></span>
            <span class="hb-pct">${pct}%</span>
          </div>`;
      }).join('');
  }

  // ── render diagnosis card ─────────────────────────────────────────────────
  function renderDiagnosis(diag) {
    const cardEl = el('inv-diagnosis');
    if (!cardEl) return;
    const conf = Math.round((diag.confidence || 0) * 100);
    const fmtList = (arr) => (arr || []).map(x =>
      '<li>' + esc(x) + '</li>'
    ).join('');
    const fmtCitations = (arr) => (arr || []).map((c, i) =>
      `<div class="citation">[${i+1}] ${esc(c.title || c.id || '')}` +
      (c.url ? ` — <a href="${esc(c.url)}" target="_blank">${esc(c.url)}</a>` : '') +
      '</div>'
    ).join('');

    cardEl.innerHTML = `
      <div class="diagnosis-card">
        <div class="dc-title">${esc(diag.primary_hypothesis)}</div>
        <div class="dc-conf">
          Confidence: ${conf}%
          <span style="display:inline-block;width:80px;height:5px;background:#1f1f1f;border-radius:2px;margin-left:8px;vertical-align:middle;">
            <span style="display:block;height:5px;width:${conf}%;background:${conf>=70?'#22c55e':conf>=40?'#f59e0b':'#ef4444'};border-radius:2px;"></span>
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
    cardEl.style.display = '';
  }

  // ── event dispatch ────────────────────────────────────────────────────────
  window.addEventListener('selene:agent_event', (e) => {
    const ev = e.detail;
    if (!ev || !ev.type) return;

    if (ev.type === 'agent_run_started') {
      // Reset state for new run
      activeRunId = ev.run_id;
      runStartTs  = ev.timestamp;
      Object.keys(toolCalls).forEach(k => delete toolCalls[k]);
      hypotheses.length = 0;
      const hdr = el('inv-header');
      if (hdr) {
        const trig = ev.trigger || {};
        const sensors = (trig.affected_sensors || []).join(', ') || '—';
        const score   = trig.score != null ? (trig.score * 100).toFixed(0) + '%' : '';
        hdr.innerHTML = `
          <div style="font-size:12px;color:#f59e0b;margin-bottom:4px;">
            <span class="dot dot-warning" style="animation:pulse-dot 1s infinite;"></span>
            Investigation in progress
          </div>
          <div style="font-size:10px;color:#737373;">
            Trigger: <span style="color:#d4d4d4;">${esc(sensors)}</span>
            ${score ? ' · score <span style="color:#d4d4d4;">' + score + '</span>' : ''}
            · detector <span style="color:#d4d4d4;">${esc(trig.detector_name||'')}</span>
          </div>`;
      }
      const diagEl = el('inv-diagnosis');
      if (diagEl) diagEl.style.display = 'none';
      showActive();
    }

    if (ev.type === 'tool_call_started') {
      toolCalls[ev.call_id] = { name: ev.tool_name, args: ev.arguments, startTs: ev.timestamp };
      renderToolLog();
    }

    if (ev.type === 'tool_call_completed') {
      const tc = toolCalls[ev.call_id] || {};
      tc.result = ev.result_summary;
      tc.error  = ev.error || null;
      toolCalls[ev.call_id] = tc;
      renderToolLog();
    }

    if (ev.type === 'hypothesis_ladder_updated') {
      hypotheses.length = 0;
      (ev.ranked || []).forEach(([id, conf]) => hypotheses.push({ id, confidence: conf }));
      renderHypotheses();
    }

    if (ev.type === 'agent_run_completed') {
      // Swap header to completed style
      const hdr = el('inv-header');
      if (hdr) {
        const existing = hdr.querySelector('div');
        if (existing) {
          existing.innerHTML = `<span class="dot dot-nominal"></span>Investigation complete`;
          existing.style.color = '#22c55e';
        }
      }
      renderDiagnosis(ev.diagnosis || {});
    }

    if (ev.type === 'agent_run_failed') {
      const hdr = el('inv-header');
      if (hdr) {
        const existing = hdr.querySelector('div');
        if (existing) {
          existing.innerHTML = `<span class="dot dot-anomaly"></span>Investigation failed: ${esc(ev.reason)}`;
          existing.style.color = '#ef4444';
        }
      }
    }
  });

  // ── AnomalyEvent feeds idle detector log ─────────────────────────────────
  window.addEventListener('selene:telemetry', (e) => {
    // no-op here — just need the handler registered
  });

  // Feed anomaly events into detector log while idle
  const origObs = window._seleneObserverPatched;
  if (!origObs) {
    window._seleneObserverPatched = true;
    // We can't directly listen to AnomalyEvents since they come via the
    // event_state pump as type="telemetry" frames. Instead watch for
    // selene:agent_event with trigger info to populate the log.
    window.addEventListener('selene:agent_event', (e) => {
      const ev = e.detail;
      if (ev && ev.type === 'agent_run_started' && ev.trigger) {
        const t = ev.trigger;
        const sensors = (t.affected_sensors || []).join(', ');
        const ts = t.timestamp ? new Date(t.timestamp).toISOString().substring(11,19) : '';
        pushDetectorLog(ts + ' ' + (t.detector_name||'detector') + ' → ' + sensors);
      }
    });
  }
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
                value=_DATA_PANEL_HTML,
                elem_id="data-panel",
            )
            investigation_panel = gr.HTML(
                value=_INVESTIGATION_HTML,
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
    demo.load(fn=None, js=_DATA_PANEL_JS)
    demo.load(fn=None, js=_INVESTIGATION_JS)

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
