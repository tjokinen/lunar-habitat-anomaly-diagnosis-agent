---
title: LHADA — Lunar Habitat Anomaly Diagnosis
emoji: 🌑
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
tags:
  - amd
  - amd-hackathon-2026
  - vllm
  - life-support
  - anomaly-detection
---

# LHADA — Lunar Habitat Anomaly Diagnosis

On-prem reasoning agent for life-support telemetry in latency-bounded environments.

Backend runs on AMD MI300X via vLLM. See the GitHub repo for the full stack.

## What it does

LHADA replays real telemetry from the [EDEN ISS 2020 dataset](https://zenodo.org/records/11485183)
— DLR's Antarctic closed-loop greenhouse, the closest public dataset to a space habitat life-support
system — and runs an LLM-driven diagnostic agent against it.

Three scenarios are supported:

| Scenario | Type | Source |
|---|---|---|
| `thermal_loop_coolant_leak` | ISS-inspired injected signature | NASA NTRS 20190029027, 20220003097 |
| `nutrient_pump_degradation` | Native EDEN ISS anomaly archetype | Rewicki et al. 2024 |
| `co2_scrubber_efficiency_drift` | Native EDEN ISS anomaly archetype | Rewicki et al. 2024 |

## Architecture

```
EDEN ISS replayer → ScenarioInjector → TelemetryBus
                                            │
                          ┌─────────────────┼─────────────────┐
                     ThresholdDetector  DampDetector      MdiDetector
                          └─────────────────┼─────────────────┘
                                            │ AnomalyEvent
                                     ReasoningAgent (Qwen 2.5 32B / vLLM)
                                            │
                                     Diagnosis + KB citations
```

## Running locally

```bash
# Start the backend (AMD droplet or local)
cd selene
poetry run selene-serve --data-path data/eden_iss/edeniss2020 --port 8000

# Start the frontend
cd frontend
BACKEND_URL=http://localhost:8000 BACKEND_WS_URL=ws://localhost:8000 python app.py
```

Then open http://localhost:7860.
