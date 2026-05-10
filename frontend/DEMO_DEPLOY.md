# Public demo deployment (Hugging Face Space, offline)

The offline demo replays a pre-recorded `thermal_loop_coolant_leak` scenario
run with no live backend or LLM dependency. Use this when you want a demo
that works regardless of droplet uptime — e.g. for a public submission.

## Files to deploy

```
app_demo.py                                   # HF Space entry point
panels.py                                     # HTML/JS panel constants
style.css                                     # dark-theme overrides
demo_recordings/thermal_loop_coolant_leak.json
requirements.txt                              # contents below
README.md                                     # the existing one already has the HF metadata block
```

## requirements.txt

```
gradio>=5.29.0
```

That's the entire dependency tree — no `httpx`, no `websockets`, no `openai`,
no `selene` package needed. Replay timing uses only `asyncio` from stdlib.

## Setting `app_file`

The README's metadata block currently sets `app_file: app.py` (live mode).
For the offline demo, change that single line to:

```yaml
app_file: app_demo.py
```

Push the four files above plus the README to the Space and the build will
boot the offline demo.

## Capturing a fresh recording

The recording embedded in the repo is from a real backend run, but you can
regenerate it any time the live system changes:

```bash
# 1. Backend running locally on :8001
selene-serve --port 8001 --data-path data/eden_iss

# 2. Capture + analyze
poetry run python selene/scripts/capture_demo_recordings.py \
    --backend http://localhost:8001 \
    --duration 90 \
    --output frontend/demo_recordings/thermal_loop_coolant_leak.json
```

The script writes the JSON file directly into the demo's location. After
recapture, redeploy the Space.

## What the user sees

1. Page loads with the 3D habitat scene, empty sensor table, and a banner
   explaining this is a pre-recorded replay.
2. Clicking **▶ Start demo** flips the scenario timeline to active and
   begins streaming the captured events at their original cadence.
3. Pressure decays, temperatures drift, valve steps, DAMP fires, and the
   investigation panel populates with tool calls and a final diagnosis card
   citing NTRS 20190029027.
4. Clicking **↺ Reset** at any point cancels the in-flight replay and
   clears the panels — the Gradio `cancels=[…]` arg is what aborts the
   running async generator.
5. After completion, the Start button re-arms as "▶ Replay again".

The clock bar shows the effective sim/wall ratio computed from the recording
(~539×, derived from telemetry timestamps) instead of the recording's
metadata field, which is sometimes stale.
