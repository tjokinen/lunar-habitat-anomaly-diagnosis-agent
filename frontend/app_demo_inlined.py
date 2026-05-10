"""LHADA — Lunar Habitat Anomaly Diagnosis Agent (offline public demo, inlined).

Self-contained Gradio app for Hugging Face Spaces deployment. Replays a
pre-recorded scenario run from ``demo_recordings/thermal_loop_coolant_leak.json``
and shows the same panels as the live app — habitat scene, sensor table with
sparklines, scenario timeline, investigation trace, final diagnosis card —
without depending on a live FastAPI backend or vLLM endpoint.

The recording was captured by ``selene/scripts/capture_demo_recordings.py``
running against a real backend; every telemetry frame and agent event is
preserved with its original wall-clock offset, so playback timing matches
what a viewer would see in the live system.

Files needed at deploy time (inlined variant)
---------------------------------------------
- app_demo_inlined.py                         (this file; HF Space ``app_file``)
- demo_recordings/thermal_loop_coolant_leak.json
- requirements.txt                            (just ``gradio``)

panels.py and style.css are inlined into this file as module-level string
constants — no other Python or CSS files needed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr
from gradio.themes import Base

# ---------------------------------------------------------------------------
# Inlined HTML/JS panel constants (originally frontend/panels.py).
# Self-contained: this file + the recording JSON is all you need on HF Spaces.
# ---------------------------------------------------------------------------

HABITAT_HTML = '\n<div id="selene-habitat-root" style="width:100%;height:440px;position:relative;background:#050508;border-radius:4px;overflow:hidden;">\n  <canvas id="selene-habitat-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;display:block;"></canvas>\n  <div id="selene-habitat-status" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-family:monospace;font-size:11px;color:#404040;pointer-events:none;">\n    initializing scene…\n  </div>\n  <div id="selene-legend" style="position:absolute;bottom:10px;left:12px;font-family:monospace;font-size:10px;color:#d4d4d4;display:flex;gap:12px;pointer-events:none;display:none;">\n    <span><span id="leg-thermal"  style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#60a5fa;margin-right:4px;vertical-align:middle;"></span>THERMAL</span>\n    <span><span id="leg-atmos"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;margin-right:4px;vertical-align:middle;"></span>ATMOS</span>\n    <span><span id="leg-nutrient" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#facc15;margin-right:4px;vertical-align:middle;"></span>NUTRIENTS</span>\n    <span><span id="leg-illum"    style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f97316;margin-right:4px;vertical-align:middle;"></span>ILLUMIN</span>\n  </div>\n  <div style="position:absolute;top:10px;right:12px;font-family:monospace;font-size:10px;color:#475569;pointer-events:none;">\n    Earth — comm delay: 2.6 s\n  </div>\n</div>\n'

HABITAT_JS = '\n() => {\n  // Allow re-init if previous attempt failed; guard against true double-init.\n  if (window._seleneHabitatRunning) return;\n  window._seleneHabitatRunning = true;\n\n  function setStatus(msg) {\n    const el = document.getElementById(\'selene-habitat-status\');\n    if (el) el.textContent = msg;\n  }\n\n  function loadScript(src, onload, onerror) {\n    if (document.querySelector(\'script[src="\' + src + \'"]\')) { onload(); return; }\n    const s = document.createElement(\'script\');\n    s.src = src;\n    s.onload = onload;\n    s.onerror = onerror || (() => setStatus(\'failed to load: \' + src));\n    document.head.appendChild(s);\n  }\n\n  // Wait for canvas AND non-zero dimensions (Gradio may render after JS fires).\n  function waitForCanvas(cb, elapsed) {\n    elapsed = elapsed || 0;\n    const root   = document.getElementById(\'selene-habitat-root\');\n    const canvas = document.getElementById(\'selene-habitat-canvas\');\n    if (root && canvas && root.getBoundingClientRect().width > 10) {\n      cb(canvas, root);\n      return;\n    }\n    if (elapsed > 12000) { setStatus(\'canvas not found\'); return; }\n    setTimeout(() => waitForCanvas(cb, elapsed + 200), 200);\n  }\n\n  setStatus(\'loading three.js…\');\n\n  function initScene() {\n    setStatus(\'waiting for canvas…\');\n    waitForCanvas((canvas, root) => {\n      setStatus(\'building scene…\');\n      const rect = root.getBoundingClientRect();\n      const W = Math.round(rect.width)  || 800;\n      const H = Math.round(rect.height) || 440;\n\n      const SUBSYSTEMS = {\n        thermal:  { color: 0x1d4ed8, emissive: 0x1e3a5f, x: -3.5 },\n        atmos:    { color: 0x15803d, emissive: 0x14532d, x: -1.0 },\n        nutrient: { color: 0x92400e, emissive: 0x451a03, x:  1.5 },\n        illum:    { color: 0x9a3412, emissive: 0x431407, x:  4.0 },\n      };\n      const STATUS_COLORS = {\n        warning: { color: 0xb45309, emissive: 0x78350f },\n        anomaly: { color: 0xb91c1c, emissive: 0x7f1d1d },\n      };\n\n      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });\n      renderer.setSize(W, H);\n      renderer.setPixelRatio(window.devicePixelRatio || 1);\n      renderer.shadowMap.enabled = true;\n\n      const scene = new THREE.Scene();\n      scene.background = new THREE.Color(0x050508);\n      scene.fog = new THREE.Fog(0x050508, 30, 80);\n\n      const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 200);\n      camera.position.set(0, 6, 18);\n      camera.lookAt(0, 0, 0);\n\n      // Minimal orbit controls (Three.js r160 dropped examples/js/).\n      const controls = (() => {\n        let down = false, lastX = 0, lastY = 0;\n        const sph = {\n          theta: Math.atan2(camera.position.x, camera.position.z),\n          phi:   Math.acos(Math.max(-1, Math.min(1, camera.position.y / camera.position.length()))),\n          r:     camera.position.length(),\n        };\n        const MIN_R = 8, MAX_R = 40, MAX_PHI = Math.PI * 0.55;\n        function apply() {\n          sph.phi = Math.max(0.1, Math.min(MAX_PHI, sph.phi));\n          sph.r   = Math.max(MIN_R, Math.min(MAX_R, sph.r));\n          camera.position.set(\n            sph.r * Math.sin(sph.phi) * Math.sin(sph.theta),\n            sph.r * Math.cos(sph.phi),\n            sph.r * Math.sin(sph.phi) * Math.cos(sph.theta)\n          );\n          camera.lookAt(0, 0, 0);\n        }\n        const el = renderer.domElement;\n        el.addEventListener(\'pointerdown\', e => { down = true; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture(e.pointerId); });\n        el.addEventListener(\'pointerup\',   () => { down = false; });\n        el.addEventListener(\'pointermove\', e => {\n          if (!down) return;\n          sph.theta -= (e.clientX - lastX) * 0.008;\n          sph.phi   -= (e.clientY - lastY) * 0.008;\n          lastX = e.clientX; lastY = e.clientY;\n          apply();\n        });\n        el.addEventListener(\'wheel\', e => { sph.r += e.deltaY * 0.04; apply(); e.preventDefault(); }, { passive: false });\n        apply();\n        return { update() { sph.theta += 0.003; apply(); } };\n      })();\n\n      scene.add(new THREE.AmbientLight(0xffffff, 0.15));\n      const sun = new THREE.DirectionalLight(0xfff5e0, 1.4);\n      sun.position.set(20, 30, 10); sun.castShadow = true; scene.add(sun);\n\n      // stars\n      const starVerts = [];\n      for (let i = 0; i < 1800; i++) {\n        const t = Math.random() * Math.PI * 2, p = Math.acos(2 * Math.random() - 1), r = 80 + Math.random() * 20;\n        starVerts.push(r*Math.sin(p)*Math.cos(t), r*Math.sin(p)*Math.sin(t), r*Math.cos(p));\n      }\n      const sg = new THREE.BufferGeometry();\n      sg.setAttribute(\'position\', new THREE.Float32BufferAttribute(starVerts, 3));\n      scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 0.18 })));\n\n      // lunar ground\n      const ground = new THREE.Mesh(new THREE.PlaneGeometry(120,120),\n        new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.9 }));\n      ground.rotation.x = -Math.PI/2; ground.position.y = -2.2; ground.receiveShadow = true;\n      scene.add(ground);\n\n      // Earth\n      const earth = new THREE.Mesh(new THREE.SphereGeometry(1.6,32,32),\n        new THREE.MeshStandardMaterial({ color: 0x1a56a0, emissive: 0x0a2040, roughness: 0.6 }));\n      earth.position.set(-22, 18, -40); scene.add(earth);\n      earth.add(new THREE.Mesh(new THREE.SphereGeometry(1.62,32,32),\n        new THREE.MeshBasicMaterial({ color: 0x2d6a2d, transparent: true, opacity: 0.35 })));\n\n      // habitat shell\n      const shell = new THREE.Mesh(new THREE.CylinderGeometry(2.2,2.2,11,32,1,true),\n        new THREE.MeshStandardMaterial({ color: 0x1e293b, transparent: true, opacity: 0.18, side: THREE.DoubleSide, roughness:0.5, metalness:0.3 }));\n      shell.rotation.z = Math.PI/2; scene.add(shell);\n      for (const xOff of [-5.5, 5.5]) {\n        const cap = new THREE.Mesh(new THREE.CircleGeometry(2.2,32),\n          new THREE.MeshStandardMaterial({ color: 0x1e293b, transparent:true, opacity:0.35, side:THREE.DoubleSide }));\n        cap.rotation.y = Math.PI/2; cap.position.x = xOff; scene.add(cap);\n      }\n      for (const xOff of [-4,-1.5,1,3.5]) {\n        const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2,0.08,8,32),\n          new THREE.MeshStandardMaterial({ color: 0x334155, metalness:0.7, roughness:0.3 }));\n        ring.rotation.y = Math.PI/2; ring.position.x = xOff; scene.add(ring);\n      }\n\n      // subsystem nodes\n      const subsystemMeshes = {}, subsystemPulse = {};\n      for (const [name, cfg] of Object.entries(SUBSYSTEMS)) {\n        const g = new THREE.Group(); g.position.x = cfg.x;\n        const mat = new THREE.MeshStandardMaterial({ color:cfg.color, emissive:cfg.emissive, emissiveIntensity:0.4, roughness:0.4, metalness:0.5 });\n        const box = new THREE.Mesh(new THREE.BoxGeometry(0.7,0.7,0.7), mat); box.castShadow=true; g.add(box);\n        const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.12,1.6,10),\n          new THREE.MeshStandardMaterial({ color:cfg.color, emissive:cfg.emissive, emissiveIntensity:0.2, metalness:0.7 }));\n        pipe.position.y = -0.9; g.add(pipe);\n        for (let i=0;i<3;i++) {\n          const s = new THREE.Mesh(new THREE.SphereGeometry(0.1,8,8), new THREE.MeshBasicMaterial({color:0xffffff}));\n          s.position.set((i-1)*0.5, 0.55+i*0.15, 0.4); g.add(s);\n        }\n        const ptLight = new THREE.PointLight(cfg.color, 0.6, 3.5); ptLight.position.set(0,0.3,0); g.add(ptLight);\n        g.userData = { name, mat, ptLight, origColor:cfg.color, origEmissive:cfg.emissive, clickable:true };\n        scene.add(g);\n        subsystemMeshes[name] = g;\n        subsystemPulse[name]  = { active:false, t:0 };\n      }\n\n      // click handler\n      const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();\n      renderer.domElement.addEventListener(\'click\', (e) => {\n        const r = renderer.domElement.getBoundingClientRect();\n        mouse.x = ((e.clientX-r.left)/r.width)*2-1;\n        mouse.y = -((e.clientY-r.top)/r.height)*2+1;\n        ray.setFromCamera(mouse, camera);\n        for (const hit of ray.intersectObjects(scene.children, true)) {\n          let o = hit.object;\n          while (o.parent && !o.userData.clickable) o = o.parent;\n          if (o.userData.clickable) {\n            window.dispatchEvent(new CustomEvent(\'selene:subsystem_selected\', { detail: o.userData.name }));\n            break;\n          }\n        }\n      });\n\n      // status updates\n      const LEG_COLORS = { thermal:\'#60a5fa\', atmos:\'#4ade80\', nutrient:\'#facc15\', illum:\'#f97316\' };\n      function setSubsystemStatus(name, status) {\n        const g = subsystemMeshes[name]; if (!g) return;\n        const { mat, ptLight, origColor, origEmissive } = g.userData;\n        const cfg = STATUS_COLORS[status];\n        if (cfg) { mat.color.setHex(cfg.color); mat.emissive.setHex(cfg.emissive); ptLight.color.setHex(cfg.color); }\n        else { mat.color.setHex(origColor); mat.emissive.setHex(origEmissive); ptLight.color.setHex(origColor); }\n        if (status === \'anomaly\') subsystemPulse[name] = { active:true, t:0 };\n        const dot = document.getElementById(\'leg-\'+name);\n        if (dot) dot.style.background = status===\'anomaly\'?\'#ef4444\':status===\'warning\'?\'#f59e0b\':(LEG_COLORS[name]||\'#888\');\n      }\n      function sensorToSub(id) {\n        if (!id) return null;\n        if (id.startsWith(\'tcs\')) return \'thermal\';\n        if (id.startsWith(\'ams\')) return \'atmos\';\n        if (id.startsWith(\'nds\')) return \'nutrient\';\n        if (id.startsWith(\'ics\')) return \'illum\';\n        return null;\n      }\n      window.addEventListener(\'selene:agent_event\', (e) => {\n        const ev = e.detail; if (!ev) return;\n        if (ev.type === \'agent_run_completed\' && ev.diagnosis) {\n          (ev.diagnosis.affected_sensors||[]).forEach(s => { const sub=sensorToSub(s); if(sub) setSubsystemStatus(sub,\'anomaly\'); });\n        }\n        if (ev.type === \'anomaly\' && ev.affected_sensors) {\n          ev.affected_sensors.forEach(s => { const sub=sensorToSub(s); if(sub) setSubsystemStatus(sub,\'warning\'); });\n        }\n      });\n\n      // hide status, show legend once scene is ready\n      const statusEl = document.getElementById(\'selene-habitat-status\');\n      if (statusEl) statusEl.style.display = \'none\';\n      const legendEl = document.getElementById(\'selene-legend\');\n      if (legendEl) legendEl.style.display = \'flex\';\n\n      // animation loop\n      const clock = new THREE.Clock();\n      (function animate() {\n        requestAnimationFrame(animate);\n        const dt = clock.getDelta();\n        for (const [name, ps] of Object.entries(subsystemPulse)) {\n          if (!ps.active) continue;\n          ps.t += dt*2.5;\n          const g = subsystemMeshes[name];\n          if (g) g.userData.mat.emissiveIntensity = 0.3 + 0.5*(0.5+0.5*Math.sin(ps.t*Math.PI));\n        }\n        earth.rotation.y += dt*0.04;\n        controls.update();\n        renderer.render(scene, camera);\n      })();\n\n      // resize\n      new ResizeObserver(() => {\n        const r2 = root.getBoundingClientRect();\n        const w = Math.round(r2.width) || 800, h = Math.round(r2.height) || 440;\n        renderer.setSize(w, h); camera.aspect = w/h; camera.updateProjectionMatrix();\n      }).observe(root);\n    });\n  }\n\n  loadScript(\n    \'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js\',\n    initScene\n  );\n}\n'

DATA_PANEL_HTML = '\n<div class="selene-panel" id="selene-data-panel">\n  <div class="clock-bar">\n    <span class="lunar-time" id="lunar-clock">--:--:--</span>\n    <span class="speed-indicator" id="speed-indicator" title="Replay speed">—×</span>\n    <span class="comm-window" id="comm-window-label">Earth link: 2.6 s RTT</span>\n  </div>\n  <div class="scenario-timeline" id="scenario-timeline">\n    <div class="st-state" id="st-state">No scenario running.</div>\n    <div class="st-bar" id="st-bar"><div class="st-fill" id="st-fill"></div></div>\n    <div class="st-meta" id="st-meta"></div>\n  </div>\n  <table class="sensor-table" id="sensor-table">\n    <thead>\n      <tr>\n        <th style="width:52%;text-align:left;font-size:9px;color:#525252;font-weight:400;padding:0 8px 4px;letter-spacing:.06em;">SENSOR</th>\n        <th style="width:26%;text-align:right;font-size:9px;color:#525252;font-weight:400;padding:0 4px 4px;letter-spacing:.06em;">VALUE</th>\n        <th style="width:22%;text-align:right;font-size:9px;color:#525252;font-weight:400;padding:0 8px 4px;letter-spacing:.06em;">TREND</th>\n      </tr>\n    </thead>\n    <tbody id="sensor-rows">\n      <tr><td colspan="3" style="color:#404040;font-size:12px;padding:20px 8px;">\n        Waiting for telemetry…\n      </td></tr>\n    </tbody>\n  </table>\n</div>\n'

DATA_PANEL_JS = '\n() => {\n  if (window._seleneDataPanelInit) return;\n  window._seleneDataPanelInit = true;\n\n  // ── Sensors to display ────────────────────────────────────────────────────\n  const SENSORS = [\n    { id: \'tcs/temp-ams_1\',       label: \'TCS Temp AMS-1\',     unit: \'°C\'   },\n    { id: \'tcs/temp-ams_2\',       label: \'TCS Temp AMS-2\',     unit: \'°C\'   },\n    { id: \'tcs/pressure-ams\',     label: \'TCS Pressure AMS\',   unit: \'bar\'  },\n    { id: \'tcs/rh-ams_1\',         label: \'TCS Humidity AMS-1\', unit: \'%\'    },\n    { id: \'ams-feg/co2-1\',        label: \'CO₂ FEG\',            unit: \'ppm\'  },\n    { id: \'ams-ses/co2-1\',        label: \'CO₂ SES\',            unit: \'ppm\'  },\n    { id: \'ams-feg/o2-1\',         label: \'O₂ FEG\',             unit: \'%\'    },\n    { id: \'nds/level-tank1\',      label: \'Nutrient Tank 1\',    unit: \'cm\'   },\n    { id: \'nds/level-tank2\',      label: \'Nutrient Tank 2\',    unit: \'cm\'   },\n    { id: \'nds/volume-tank1\',     label: \'NDS Volume 1\',       unit: \'L\'    },\n    { id: \'ics/par-1\',            label: \'ICS PAR-1\',          unit: \'µmol\' },\n  ];\n\n  // sparkline history: sensorId → [value, ...]  (max 60 points = 5-min cadence × 5 h)\n  const HISTORY_LEN = 60;\n  const history = {};\n  const sensorStatus = {};  // sensorId → \'nominal\'|\'warning\'|\'anomaly\'\n  SENSORS.forEach(s => { history[s.id] = []; sensorStatus[s.id] = \'nominal\'; });\n\n  // ── Lunar clock ────────────────────────────────────────────────────────────\n  // Drive from telemetry timestamps; fall back to wall clock offset.\n  let lastTelemetryTs = null;\n  let lastWallMs = Date.now();\n\n  function fmtTime(isoOrDate) {\n    const d = typeof isoOrDate === \'string\' ? new Date(isoOrDate) : isoOrDate;\n    if (isNaN(d)) return \'--:--:--\';\n    return d.toISOString().substring(11, 19);\n  }\n\n  // simNow() returns the current simulated time in ms (Unix epoch), derived\n  // from the last telemetry timestamp + wall-clock elapsed since then. This\n  // lets the comm-window cycle and indicators speed up with --speed N.\n  function simNowMs() {\n    if (!lastTelemetryTs) return null;\n    return new Date(lastTelemetryTs).getTime() + (Date.now() - lastWallMs);\n  }\n\n  function tickClock() {\n    const clockEl = document.getElementById(\'lunar-clock\');\n    if (!clockEl) return;\n    const sim = simNowMs();\n    if (sim !== null) clockEl.textContent = fmtTime(new Date(sim));\n    updateScenarioTimeline();\n  }\n  setInterval(tickClock, 1000);\n\n  // ── Scenario timeline ──────────────────────────────────────────────────\n  // ground_truth = { scenario_id, start_time, end_time, affected_sensors, description }\n  // OR null if no scenario is active.\n  let scenarioGT = null;\n  let scenarioName = null;\n  let firstDetectionMs = null;   // wall-clock-aligned sim time of first agent_run_started\n\n  function fmtDur(ms) {\n    if (ms == null || isNaN(ms)) return \'—\';\n    const sign = ms < 0 ? \'-\' : \'\';\n    const s = Math.abs(Math.round(ms / 1000));\n    const h = Math.floor(s / 3600);\n    const m = Math.floor((s % 3600) / 60);\n    const sec = s % 60;\n    if (h > 0) return sign + h + \'h \' + m + \'m\';\n    if (m > 0) return sign + m + \'m \' + sec + \'s\';\n    return sign + sec + \'s\';\n  }\n\n  function setTimelineClass(cls) {\n    const el = document.getElementById(\'scenario-timeline\');\n    if (!el) return;\n    el.className = \'scenario-timeline \' + cls;\n  }\n\n  function updateScenarioTimeline() {\n    const stateEl = document.getElementById(\'st-state\');\n    const fillEl  = document.getElementById(\'st-fill\');\n    const metaEl  = document.getElementById(\'st-meta\');\n    if (!stateEl || !fillEl || !metaEl) return;\n\n    if (!scenarioGT) {\n      setTimelineClass(\'st-idle\');\n      stateEl.innerHTML = \'<span class="st-label">No scenario running.</span>\' +\n                          \'<span class="st-phase">Monitoring nominal telemetry.</span>\';\n      fillEl.style.width = \'0%\';\n      metaEl.innerHTML = \'\';\n      return;\n    }\n\n    const start = new Date(scenarioGT.start_time).getTime();\n    const end   = new Date(scenarioGT.end_time).getTime();\n    const now   = simNowMs();\n    if (now == null || isNaN(start) || isNaN(end) || end <= start) {\n      setTimelineClass(\'st-idle\');\n      stateEl.innerHTML = \'<span class="st-label">\' + esc(scenarioName || scenarioGT.scenario_id) + \'</span>\' +\n                          \'<span class="st-phase">waiting for telemetry…</span>\';\n      fillEl.style.width = \'0%\';\n      metaEl.innerHTML = \'\';\n      return;\n    }\n\n    const total = end - start;\n    let phase, pct, phaseLabel;\n    if (now < start) {\n      phase = \'st-pre\';\n      pct = 0;\n      phaseLabel = \'Pre-onset · starts in \' + fmtDur(start - now);\n    } else if (now < end) {\n      phase = \'st-active\';\n      pct = Math.max(0, Math.min(100, ((now - start) / total) * 100));\n      phaseLabel = \'Anomaly active · \' + fmtDur(now - start) + \' / \' + fmtDur(total);\n    } else {\n      phase = \'st-post\';\n      pct = 100;\n      phaseLabel = \'Anomaly window ended · \' + fmtDur(now - end) + \' ago\';\n    }\n\n    setTimelineClass(phase);\n    stateEl.innerHTML = \'<span class="st-label">\' + esc(scenarioName || scenarioGT.scenario_id) + \'</span>\' +\n                        \'<span class="st-phase">\' + esc(phaseLabel) + \'</span>\';\n    fillEl.style.width = pct.toFixed(1) + \'%\';\n\n    let detectionLine = \'Detection: <span style="color:#737373;">—</span>\';\n    if (firstDetectionMs != null) {\n      const lag = firstDetectionMs - start;\n      const lagLabel = lag >= 0\n        ? \'+\' + fmtDur(lag) + \' after onset\'\n        : fmtDur(lag) + \' before onset\';\n      detectionLine = \'Detection: <span style="color:#f59e0b;">\' +\n                      esc(fmtTime(new Date(firstDetectionMs))) + \'</span> · \' +\n                      esc(lagLabel);\n    }\n\n    metaEl.innerHTML =\n      \'<span>Onset \' + esc(fmtTime(new Date(start))) +\n        \' → end \' + esc(fmtTime(new Date(end))) + \'</span>\' +\n      \'<span>\' + detectionLine + \'</span>\';\n  }\n\n  function esc(s) {\n    return String(s).replace(/&/g,\'&amp;\').replace(/</g,\'&lt;\').replace(/>/g,\'&gt;\');\n  }\n\n  window.addEventListener(\'selene:scenario\', (e) => {\n    const d = e.detail || {};\n    if (d.active && d.ground_truth) {\n      scenarioGT = d.ground_truth;\n      scenarioName = d.scenario_name || d.scenario_id || null;\n      firstDetectionMs = null;   // new scenario → reset detection lag\n    } else {\n      scenarioGT = null;\n      scenarioName = null;\n      firstDetectionMs = null;\n    }\n    updateScenarioTimeline();\n  });\n\n  window.addEventListener(\'selene:first-detection\', (e) => {\n    const ts = e.detail && e.detail.timestamp;\n    if (!ts) return;\n    if (firstDetectionMs == null) {\n      firstDetectionMs = new Date(ts).getTime();\n      updateScenarioTimeline();\n    }\n  });\n\n\n  // ── Speed indicator (set by Gradio via window.__seleneSetSpeed) ──────────\n  window.__seleneSetSpeed = function(mult) {\n    const el = document.getElementById(\'speed-indicator\');\n    if (!el) return;\n    el.textContent = (mult === null || mult === undefined) ? \'∞×\' : (mult + \'×\');\n  };\n\n  // ── Sparkline ─────────────────────────────────────────────────────────────\n  function drawSparkline(canvas, values) {\n    const ctx = canvas.getContext(\'2d\');\n    const W = canvas.width, H = canvas.height;\n    ctx.clearRect(0, 0, W, H);\n    if (values.length < 2) return;\n\n    const min = Math.min(...values), max = Math.max(...values);\n    const range = max - min || 1;\n\n    ctx.beginPath();\n    ctx.strokeStyle = \'#3b82f6\';\n    ctx.lineWidth = 1.5;\n    values.forEach((v, i) => {\n      const x = (i / (values.length - 1)) * W;\n      const y = H - ((v - min) / range) * (H - 2) - 1;\n      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);\n    });\n    ctx.stroke();\n  }\n\n  // ── Build / update sensor rows ─────────────────────────────────────────────\n  let rowsBuilt = false;\n\n  function buildRows() {\n    const tbody = document.getElementById(\'sensor-rows\');\n    if (!tbody) return;\n    tbody.innerHTML = \'\';\n    SENSORS.forEach(s => {\n      const tr = document.createElement(\'tr\');\n      tr.id = \'row-\' + s.id.replace(/\\//g, \'-\').replace(/_/g, \'-\');\n      tr.innerHTML = `\n        <td style="padding:5px 8px;">\n          <span class="dot dot-nominal" id="dot-${s.id.replace(/\\//g,\'-\').replace(/_/g,\'-\')}"></span>\n          <span class="sensor-name">${s.label}</span>\n        </td>\n        <td style="padding:5px 4px;text-align:right;">\n          <span class="sensor-value" id="val-${s.id.replace(/\\//g,\'-\').replace(/_/g,\'-\')}">—</span>\n          <span class="sensor-unit">${s.unit}</span>\n        </td>\n        <td style="padding:3px 8px 3px 4px;text-align:right;vertical-align:middle;">\n          <canvas id="spk-${s.id.replace(/\\//g,\'-\').replace(/_/g,\'-\')}"\n                  width="72" height="22"\n                  style="display:inline-block;vertical-align:middle;"></canvas>\n        </td>`;\n      tbody.appendChild(tr);\n    });\n    rowsBuilt = true;\n  }\n\n  function safeId(sensorId) { return sensorId.replace(/\\//g,\'-\').replace(/_/g,\'-\'); }\n\n  function updateRow(sensorId, value) {\n    if (!rowsBuilt) buildRows();\n    const sid = safeId(sensorId);\n    const valEl = document.getElementById(\'val-\' + sid);\n    if (valEl) valEl.textContent = typeof value === \'number\' ? value.toFixed(2) : value;\n\n    const hist = history[sensorId];\n    if (hist) {\n      hist.push(value);\n      if (hist.length > HISTORY_LEN) hist.shift();\n      const canvas = document.getElementById(\'spk-\' + sid);\n      if (canvas) drawSparkline(canvas, hist);\n    }\n  }\n\n  function setRowStatus(sensorId, status) {\n    sensorStatus[sensorId] = status;\n    const sid = safeId(sensorId);\n    const dot = document.getElementById(\'dot-\' + sid);\n    if (!dot) return;\n    dot.className = \'dot dot-\' + status;\n  }\n\n  // ── Telemetry event handler ────────────────────────────────────────────────\n  window.addEventListener(\'selene:telemetry\', (e) => {\n    const frame = e.detail;\n    if (!frame) return;\n\n    // Update clock\n    if (frame.timestamp) {\n      lastTelemetryTs = frame.timestamp;\n      lastWallMs = Date.now();\n      const clockEl = document.getElementById(\'lunar-clock\');\n      if (clockEl) clockEl.textContent = fmtTime(frame.timestamp);\n    }\n\n    if (!rowsBuilt) buildRows();\n\n    // Update each displayed sensor\n    const readings = frame.readings || {};\n    SENSORS.forEach(s => {\n      const r = readings[s.id];\n      if (r === undefined || r === null) return;\n      // Wire format is the full SensorReading object; pull the numeric value.\n      const v = (typeof r === \'object\') ? r.value : r;\n      if (v !== undefined && v !== null) updateRow(s.id, v);\n    });\n  });\n\n  // ── Agent event → sensor status ───────────────────────────────────────────\n  window.addEventListener(\'selene:agent_event\', (e) => {\n    const ev = e.detail; if (!ev) return;\n    if (ev.affected_sensors) {\n      ev.affected_sensors.forEach(sid => setRowStatus(sid, \'warning\'));\n    }\n    if (ev.type === \'agent_run_completed\' && ev.diagnosis) {\n      (ev.diagnosis.affected_sensors || []).forEach(sid => setRowStatus(sid, \'anomaly\'));\n    }\n  });\n\n  // Build rows immediately so the panel isn\'t blank.\n  function waitAndBuild() {\n    if (document.getElementById(\'sensor-rows\')) { buildRows(); return; }\n    setTimeout(waitAndBuild, 200);\n  }\n  waitAndBuild();\n}\n'

INVESTIGATION_HTML = '\n<div class="selene-panel" id="selene-inv-panel" style="min-height:280px;">\n  <div class="trace-header">Investigation trace</div>\n\n  <!-- idle banner shown only while there are zero runs -->\n  <div id="inv-idle" style="display:flex;align-items:center;gap:8px;color:#404040;font-size:12px;padding:8px 0 12px;">\n    <span class="dot dot-nominal" style="animation:pulse-dot 2s infinite;"></span>\n    Monitoring nominal — no active investigation.\n  </div>\n\n  <!-- Stack of run blocks. Newest investigation sits at the top; older\n       runs remain visible below as collapsed cards. -->\n  <div id="inv-runs" style="max-height:520px;overflow-y:auto;"></div>\n</div>\n'

INVESTIGATION_JS = '\n() => {\n  if (window._seleneInvInit) return;\n  window._seleneInvInit = true;\n\n  // ── per-run state ────────────────────────────────────────────────────────\n  // Newest run goes to the front of the list. Each entry holds everything\n  // needed to re-render that run\'s block without reading the DOM.\n  // run = { runId, startTs, trigger, status, toolCalls{call_id→tc},\n  //         toolOrder[call_id...], hypotheses[], diagnosis, failReason }\n  const runs = [];\n  function getRun(runId) { return runs.find(r => r.runId === runId); }\n\n  // ── helpers ──────────────────────────────────────────────────────────────\n  function el(id) { return document.getElementById(id); }\n\n  function esc(s) {\n    return String(s).replace(/&/g,\'&amp;\').replace(/</g,\'&lt;\').replace(/>/g,\'&gt;\');\n  }\n\n  function elapsed(startTs, ts) {\n    if (!startTs || !ts) return \'\';\n    const dt = Math.round((new Date(ts) - new Date(startTs)) / 1000);\n    const m = Math.floor(dt / 60), s = dt % 60;\n    return \'[+\' + (m ? m + \':\' + String(s).padStart(2,\'0\') : s + \'s\') + \']\';\n  }\n\n  function shortArgs(args) {\n    try {\n      const s = JSON.stringify(args);\n      if (s === undefined) return \'{}\';\n      return s.length > 80 ? s.slice(0, 77) + \'…\' : s;\n    } catch { return \'{}\'; }\n  }\n\n  function fmtClock(ts) {\n    if (!ts) return \'\';\n    try { return new Date(ts).toISOString().substring(11,19); }\n    catch { return \'\'; }\n  }\n\n  // ── header HTML for one run ──────────────────────────────────────────────\n  function headerHTML(run) {\n    const trig = run.trigger || {};\n    const sensors = (trig.affected_sensors || []).join(\', \') || \'—\';\n    const score   = trig.score != null ? trig.score.toFixed(2) : \'\';\n    const det     = trig.detector_name || \'\';\n    const ts      = fmtClock(run.startTs);\n\n    let title, color;\n    if (run.status === \'running\') {\n      title = `<span class="dot dot-warning" style="animation:pulse-dot 1s infinite;"></span>Investigation in progress`;\n      color = \'#f59e0b\';\n    } else if (run.status === \'completed\') {\n      title = `<span class="dot dot-anomaly"></span>Investigation complete`;\n      color = \'#ef4444\';\n    } else if (run.status === \'failed\') {\n      title = `<span class="dot dot-anomaly"></span>Investigation failed: ${esc(run.failReason || \'\')}`;\n      color = \'#ef4444\';\n    } else {\n      title = \'Investigation\';\n      color = \'#a3a3a3\';\n    }\n\n    return `\n      <div style="font-size:12px;color:${color};margin-bottom:4px;">${title}</div>\n      <div style="font-size:10px;color:#737373;">\n        ${ts ? \'<span style="color:#525252;margin-right:8px;">\' + esc(ts) + \'</span>\' : \'\'}\n        Trigger: <span style="color:#d4d4d4;">${esc(sensors)}</span>\n        ${score ? \' · score <span style="color:#d4d4d4;">\' + score + \'</span>\' : \'\'}\n        ${det ? \' · detector <span style="color:#d4d4d4;">\' + esc(det) + \'</span>\' : \'\'}\n      </div>`;\n  }\n\n  // ── tool log HTML for one run ────────────────────────────────────────────\n  function toolLogHTML(run) {\n    if (!run.toolOrder || run.toolOrder.length === 0) return \'\';\n    const lines = run.toolOrder.map(cid => {\n      const tc = run.toolCalls[cid];\n      if (!tc) return \'\';\n      const done = tc.result !== undefined || tc.error;\n      const result = tc.error\n        ? \'<span style="color:#ef4444;">error: \' + esc(tc.error) + \'</span>\'\n        : tc.result\n          ? \'<span style="color:#ef4444;">\' + esc(tc.result) + \'</span>\'\n          : \'<span style="color:#525252;">…</span>\';\n      return `\n        <div class="tool-call">\n          <span style="color:#475569;margin-right:6px;">${esc(elapsed(run.startTs, tc.startTs))}</span>\n          <span class="tc-name">${esc(tc.name || \'?\')}</span>\n          <span class="tc-args">${esc(shortArgs(tc.args))}</span>\n          ${done ? \'<span class="tc-result"> → \' + result + \'</span>\' : \'\'}\n        </div>`;\n    }).join(\'\');\n    return \'<div class="tool-log">\' + lines + \'</div>\';\n  }\n\n  // ── hypothesis ladder HTML for one run ───────────────────────────────────\n  function hypothesesHTML(run) {\n    if (!run.hypotheses || run.hypotheses.length === 0) return \'\';\n    return \'<div class="trace-header" style="margin-bottom:4px;">Hypotheses</div>\' +\n      run.hypotheses.map(h => {\n        const pct = Math.round(h.confidence * 100);\n        return `\n          <div class="hypothesis-bar">\n            <span class="hb-label" title="${esc(h.id)}">${esc(h.id)}</span>\n            <span class="hb-track"><span class="hb-fill" style="width:${pct}%;"></span></span>\n            <span class="hb-pct">${pct}%</span>\n          </div>`;\n      }).join(\'\');\n  }\n\n  // ── diagnosis card HTML for one run ──────────────────────────────────────\n  function diagnosisHTML(run) {\n    const diag = run.diagnosis;\n    if (!diag) return \'\';\n    const conf = Math.round((diag.confidence || 0) * 100);\n    const fmtList = (arr) => (arr || []).map(x => \'<li>\' + esc(x) + \'</li>\').join(\'\');\n    const fmtCitations = (arr) => (arr || []).map((c, i) =>\n      `<div class="citation">[${i+1}] ${esc(c.title || c.id || \'\')}` +\n      (c.url ? ` — <a href="${esc(c.url)}" target="_blank">${esc(c.url)}</a>` : \'\') +\n      \'</div>\'\n    ).join(\'\');\n\n    return `\n      <div class="diagnosis-card">\n        <div class="dc-title">${esc(diag.primary_hypothesis || \'—\')}</div>\n        <div class="dc-conf">\n          Confidence: ${conf}%\n          <span style="display:inline-block;width:80px;height:5px;background:#1f1f1f;border-radius:2px;margin-left:8px;vertical-align:middle;">\n            <span style="display:block;height:5px;width:${conf}%;background:${conf>=70?\'#ef4444\':conf>=40?\'#f59e0b\':\'#737373\'};border-radius:2px;"></span>\n          </span>\n        </div>\n        ${diag.matched_failure_modes && diag.matched_failure_modes.length ? `\n          <div class="dc-section">Matched failure modes</div>\n          <ul>${fmtList(diag.matched_failure_modes)}</ul>` : \'\'}\n        ${diag.supporting_evidence && diag.supporting_evidence.length ? `\n          <div class="dc-section">Supporting evidence</div>\n          <ul>${fmtList(diag.supporting_evidence)}</ul>` : \'\'}\n        ${diag.recommended_actions && diag.recommended_actions.length ? `\n          <div class="dc-section">Recommended actions</div>\n          <ul>${fmtList(diag.recommended_actions)}</ul>` : \'\'}\n        ${diag.citations && diag.citations.length ? `\n          <div class="dc-section">References</div>\n          ${fmtCitations(diag.citations)}` : \'\'}\n      </div>`;\n  }\n\n  // ── render the full stack of run blocks ──────────────────────────────────\n  function render() {\n    const root = el(\'inv-runs\');\n    if (!root) return;\n    const idle = el(\'inv-idle\');\n    if (idle) idle.style.display = runs.length === 0 ? \'\' : \'none\';\n\n    root.innerHTML = runs.map((run, idx) => {\n      const cls = run.status === \'running\'\n        ? \'inv-run-block inv-run-active\'\n        : \'inv-run-block inv-run-past\';\n      return `\n        <div class="${cls}" data-run="${esc(run.runId)}">\n          ${headerHTML(run)}\n          ${toolLogHTML(run)}\n          ${hypothesesHTML(run)}\n          ${diagnosisHTML(run)}\n        </div>`;\n    }).join(\'\');\n  }\n\n  // ── event dispatch ───────────────────────────────────────────────────────\n  window.addEventListener(\'selene:agent_event\', (e) => {\n    const ev = e.detail;\n    if (!ev || !ev.type) return;\n\n    if (ev.type === \'agent_run_started\') {\n      // Mark any still-running run as failed (lost) before pushing new one.\n      runs.forEach(r => { if (r.status === \'running\') r.status = \'failed\'; });\n      runs.unshift({\n        runId: ev.run_id,\n        startTs: ev.timestamp,\n        trigger: ev.trigger || {},\n        status: \'running\',\n        toolCalls: {},\n        toolOrder: [],\n        hypotheses: [],\n        diagnosis: null,\n        failReason: null,\n      });\n      // Cap retained history.\n      if (runs.length > 12) runs.length = 12;\n      // Notify the data panel so the scenario timeline can show\n      // detection lag (it dedups internally).\n      window.dispatchEvent(new CustomEvent(\'selene:first-detection\',\n        { detail: { timestamp: ev.timestamp } }));\n      render();\n    }\n\n    if (ev.type === \'tool_call_started\') {\n      const run = getRun(ev.run_id);\n      if (!run) return;\n      if (!(ev.call_id in run.toolCalls)) run.toolOrder.push(ev.call_id);\n      run.toolCalls[ev.call_id] = {\n        name: ev.tool_name, args: ev.arguments, startTs: ev.timestamp,\n      };\n      render();\n    }\n\n    if (ev.type === \'tool_call_completed\') {\n      const run = getRun(ev.run_id);\n      if (!run) return;\n      const tc = run.toolCalls[ev.call_id] || {};\n      // Fallback: if started was missed (e.g. WS reconnect mid-run),\n      // completed still carries name/args.\n      if (tc.name === undefined)    tc.name    = ev.tool_name;\n      if (tc.args === undefined)    tc.args    = ev.arguments;\n      if (tc.startTs === undefined) tc.startTs = ev.timestamp;\n      tc.result = ev.result_summary;\n      tc.error  = ev.error || null;\n      if (!(ev.call_id in run.toolCalls)) run.toolOrder.push(ev.call_id);\n      run.toolCalls[ev.call_id] = tc;\n      render();\n    }\n\n    if (ev.type === \'hypothesis_ladder_updated\') {\n      const run = getRun(ev.run_id);\n      if (!run) return;\n      run.hypotheses = (ev.ranked || []).map(([id, conf]) => ({ id, confidence: conf }));\n      render();\n    }\n\n    if (ev.type === \'agent_run_completed\') {\n      const run = getRun(ev.run_id);\n      if (!run) return;\n      run.status = \'completed\';\n      run.diagnosis = ev.diagnosis || null;\n      render();\n    }\n\n    if (ev.type === \'agent_run_failed\') {\n      const run = getRun(ev.run_id);\n      if (!run) return;\n      run.status = \'failed\';\n      run.failReason = ev.reason;\n      render();\n    }\n  });\n}\n'

OBSERVER_JS = "\n() => {\n  let mutCount = 0, dispatchCount = 0, parseFails = 0;\n  let lastSeq = -1;\n  window.__seleneDebug = function() {\n    console.log('[selene] mutations=' + mutCount +\n                ' dispatched=' + dispatchCount +\n                ' parseFails=' + parseFails +\n                ' lastSeq=' + lastSeq);\n  };\n\n  function dispatch(raw) {\n    try {\n      const data = JSON.parse(raw || '{}');\n      if (!data || !data.type) return;\n      if (data._seq !== undefined) lastSeq = data._seq;\n      if (data.type === 'telemetry') {\n        window.dispatchEvent(new CustomEvent('selene:telemetry', { detail: data.frame }));\n        dispatchCount++;\n      } else if (data.type === 'agent_event') {\n        window.dispatchEvent(new CustomEvent('selene:agent_event', { detail: data.event }));\n        dispatchCount++;\n      }\n    } catch(_) { parseFails++; }\n  }\n\n  function attachSeleneObserver() {\n    const root = document.querySelector('#event-state');\n    if (!root) { setTimeout(attachSeleneObserver, 200); return; }\n    const ta = root.querySelector('textarea') || root.querySelector('input');\n    if (!ta) { setTimeout(attachSeleneObserver, 200); return; }\n    console.log('[selene] observer attached to', ta.tagName, '#event-state');\n\n    // 1. Poll the textarea value — most robust across Gradio versions, since\n    //    Gradio sets .value programmatically (no native input event fires).\n    let lastVal = '';\n    setInterval(() => {\n      if (ta.value !== lastVal) {\n        lastVal = ta.value;\n        mutCount++;\n        dispatch(ta.value);\n      }\n    }, 20);\n\n    // 2. Belt-and-suspenders: also watch for DOM mutations (in case Gradio\n    //    re-renders the textarea node).\n    const obs = new MutationObserver(() => {\n      if (ta.value !== lastVal) {\n        lastVal = ta.value;\n        mutCount++;\n        dispatch(ta.value);\n      }\n    });\n    obs.observe(root, { childList: true, subtree: true, characterData: true, attributes: true });\n  }\n  attachSeleneObserver();\n\n  // ── Scenario-state observer (separate textbox, fires `selene:scenario`) ──\n  function dispatchScenario(raw) {\n    if (!raw) return;\n    try {\n      const data = JSON.parse(raw);\n      window.dispatchEvent(new CustomEvent('selene:scenario', { detail: data }));\n    } catch(_) {}\n  }\n  function attachScenarioObserver() {\n    const root = document.querySelector('#scenario-state');\n    if (!root) { setTimeout(attachScenarioObserver, 200); return; }\n    const ta = root.querySelector('textarea') || root.querySelector('input');\n    if (!ta) { setTimeout(attachScenarioObserver, 200); return; }\n    console.log('[selene] scenario observer attached');\n    let lastVal = '';\n    setInterval(() => {\n      if (ta.value !== lastVal) { lastVal = ta.value; dispatchScenario(ta.value); }\n    }, 200);\n    new MutationObserver(() => {\n      if (ta.value !== lastVal) { lastVal = ta.value; dispatchScenario(ta.value); }\n    }).observe(root, { childList: true, subtree: true, characterData: true, attributes: true });\n  }\n  attachScenarioObserver();\n}\n"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("selene.demo")

_HERE = Path(__file__).parent
_CSS = "/* ── Selene dark theme ── */\n:root {\n  --background-fill-primary:    #0a0a0a;\n  --background-fill-secondary:  #141414;\n  --background-fill-tertiary:   #1a1a1a;\n  --body-text-color:            #d4d4d4;\n  --body-text-color-subdued:    #737373;\n  --border-color-primary:       #262626;\n  --border-color-accent:        #404040;\n  --color-accent:               #3b82f6;\n  --color-accent-soft:          #1e3a5f;\n  --color-warning:              #f59e0b;\n  --color-error:                #ef4444;\n  --color-nominal:              #22c55e;\n  --block-background-fill:      #141414;\n  --block-border-color:         #262626;\n  --block-label-background-fill: #1a1a1a;\n  --block-label-text-color:     #737373;\n  --block-title-text-color:     #d4d4d4;\n  --input-background-fill:      #1a1a1a;\n  --button-primary-background-fill:        #1d4ed8;\n  --button-primary-background-fill-hover:  #2563eb;\n  --button-primary-text-color:             #ffffff;\n  --button-secondary-background-fill:      #262626;\n  --button-secondary-background-fill-hover:#404040;\n  --button-secondary-text-color:           #d4d4d4;\n  --shadow-drop:                none;\n  --shadow-drop-lg:             none;\n}\n\n/* Hide Gradio chrome */\nfooter { display: none !important; }\n.gradio-container > .main > .wrap > .md { display: none; }\n\n/* Panel base */\n.selene-panel {\n  background: var(--background-fill-secondary);\n  border: 1px solid var(--border-color-primary);\n  border-radius: 4px;\n  padding: 12px;\n  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', ui-monospace, monospace;\n  color: var(--body-text-color);\n  min-height: 200px;\n}\n\n/* Sensor value table */\n.sensor-table { width: 100%; border-collapse: collapse; font-size: 12px; }\n.sensor-table tr { border-bottom: 1px solid #1f1f1f; }\n.sensor-table td { padding: 5px 8px; }\n.sensor-table .sensor-name { color: #737373; text-transform: uppercase; letter-spacing: 0.05em; font-size: 10px; }\n.sensor-table .sensor-value { text-align: right; font-size: 15px; font-weight: 600; color: #f5f5f5; }\n.sensor-table .sensor-unit  { text-align: right; color: #525252; font-size: 10px; padding-left: 4px; }\n\n/* Status dots */\n.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }\n.dot-nominal  { background: var(--color-nominal); }\n.dot-warning  { background: var(--color-warning); }\n.dot-anomaly  { background: var(--color-error);  animation: pulse-dot 1s infinite; }\n\n@keyframes pulse-dot {\n  0%, 100% { opacity: 1; }\n  50%       { opacity: 0.3; }\n}\n\n/* Investigation trace */\n.trace-header  { font-size: 11px; color: #737373; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }\n.tool-call     { background: #0f172a; border-left: 2px solid #3b82f6; padding: 6px 10px; margin: 4px 0; font-size: 11px; border-radius: 0 3px 3px 0; }\n.tool-call .tc-name { color: #60a5fa; font-weight: 600; }\n.tool-call .tc-args { color: #94a3b8; margin-left: 8px; }\n.tool-call .tc-result{ color: #4ade80; margin-left: 8px; }\n\n.hypothesis-bar { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 11px; }\n.hypothesis-bar .hb-label { flex: 1; color: #d4d4d4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }\n.hypothesis-bar .hb-track { flex: 2; background: #1f1f1f; border-radius: 2px; height: 6px; }\n.hypothesis-bar .hb-fill  { height: 6px; background: #3b82f6; border-radius: 2px; transition: width 0.4s ease; }\n.hypothesis-bar .hb-pct   { width: 36px; text-align: right; color: #737373; }\n\n.diagnosis-card { background: #0c1a2e; border: 1px solid #1e3a5f; border-radius: 4px; padding: 12px; margin-top: 8px; }\n.diagnosis-card .dc-title { font-size: 14px; font-weight: 700; color: #f0f9ff; margin-bottom: 6px; }\n.diagnosis-card .dc-conf  { color: #60a5fa; font-size: 12px; margin-bottom: 8px; }\n.diagnosis-card .dc-section { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin: 8px 0 3px; }\n.diagnosis-card li { font-size: 12px; color: #cbd5e1; margin: 2px 0 2px 12px; }\n.citation { font-size: 10px; color: #475569; margin: 2px 0 2px 12px; }\n.citation a { color: #3b82f6; text-decoration: none; }\n\n/* Clock bar */\n.clock-bar { display: flex; justify-content: space-between; align-items: center;\n             border-bottom: 1px solid #1f1f1f; padding-bottom: 8px; margin-bottom: 10px; }\n.lunar-time { font-size: 22px; font-weight: 700; letter-spacing: 0.05em; color: #f5f5f5; }\n.comm-window{ font-size: 11px; color: #737373; }\n\n/* Scenario controls row */\n.controls-row { display: flex; gap: 8px; align-items: center; padding: 8px 0; }\n\n/* Hidden hidden event-state textbox: rendered to DOM (so the observer can\n   read it), but invisible and out of layout flow. */\n.selene-hidden { position: absolute !important; left: -9999px !important;\n                 width: 1px !important; height: 1px !important;\n                 opacity: 0 !important; pointer-events: none !important; }\n\n/* Speed indicator in the clock bar */\n.speed-indicator { font-size: 11px; color: #a3a3a3; font-variant-numeric: tabular-nums;\n                   padding: 2px 6px; border: 1px solid #262626; border-radius: 3px;\n                   margin: 0 8px; }\n\n/* Investigation run blocks (active + history) */\n.inv-run-block      { padding: 10px 0; border-top: 1px solid #1f1f1f; }\n.inv-run-block:first-child { border-top: none; padding-top: 4px; }\n.inv-run-active     { /* current run — full opacity */ }\n.inv-run-past       { opacity: 0.72; }\n.inv-run-past:hover { opacity: 1.0; transition: opacity 0.2s ease; }\n.inv-run-block .tool-log { max-height: 160px; overflow-y: auto; margin: 6px 0; }\n\n/* Scenario timeline (between clock-bar and sensor table) */\n.scenario-timeline { padding: 6px 0 8px; border-bottom: 1px solid #1f1f1f; margin-bottom: 8px; }\n.scenario-timeline .st-state { font-size: 11px; color: #a3a3a3; letter-spacing: 0.04em; margin-bottom: 5px;\n                               display: flex; justify-content: space-between; align-items: center; }\n.scenario-timeline .st-state .st-label { color: #d4d4d4; font-weight: 600; }\n.scenario-timeline .st-state .st-phase { color: #737373; }\n.scenario-timeline .st-bar { height: 4px; background: #1a1a1a; border-radius: 2px; overflow: hidden; }\n.scenario-timeline .st-fill { height: 100%; width: 0%; background: #3b82f6; transition: width 0.6s ease; }\n.scenario-timeline.st-pre   .st-fill { background: #525252; }\n.scenario-timeline.st-active .st-fill { background: #f59e0b; }\n.scenario-timeline.st-post  .st-fill { background: #22c55e; }\n.scenario-timeline .st-meta { font-size: 10px; color: #525252; margin-top: 5px;\n                              display: flex; justify-content: space-between; }\n.scenario-timeline.st-idle .st-bar,\n.scenario-timeline.st-idle .st-meta { display: none; }\n"
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
    gr.HTML(value=f"<style>{_CSS}</style>")

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
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
    launch_params = inspect.signature(demo.launch).parameters
    if "css" in launch_params:
        launch_kwargs["css"] = _CSS
    if "theme" in launch_params:
        launch_kwargs["theme"] = Base()
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
