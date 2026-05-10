# LHADA — Lunar Habitat Anomaly Diagnosis Agent

An on-prem reasoning agent for life-support telemetry in latency-bounded
environments. When the round-trip to Earth is too slow for the diagnostic loop,
the diagnosis has to happen locally. LHADA detects anomalies in real
space-analog telemetry, ranks hypotheses against a curated NASA-cited failure
mode knowledge base, and produces structured diagnoses with full evidence
traces — all on a single on-prem GPU node.

Built for the **AMD Developer Cloud Hackathon 2026** (Track 1: AI Agents &
Agentic Workflows).

---

## What's real, what's simulated

This distinction is load-bearing. State it clearly so reviewers don't have to
ask.

**Real and verified**

- **EDEN ISS 2020 Telemetry Dataset** — public, peer-reviewed, downloadable
  from Zenodo ([record 11485183](https://zenodo.org/records/11485183)).
  97 sensors across 5 subsystems (atmosphere, nutrient delivery, illumination,
  thermal control, plus FEG/SES splits) recorded at the German Aerospace
  Center (DLR) Antarctic greenhouse next to Neumayer III polar station,
  2018–2021.
- **DAMP detector** — Discord-Aware Matrix Profile via the
  [`stumpy`](https://stumpy.readthedocs.io) library. Published, validated
  unsupervised anomaly detection method, benchmarked specifically on EDEN ISS
  data in [Rewicki et al. 2024](https://arxiv.org/abs/2406.09825).
- **Failure mode knowledge base** — five entries hand-curated from NASA NTRS,
  ESA documents, and published ECLSS literature, each with full citations.
- **NASA NTRS references** — every coolant-leak diagnosis cites
  [NTRS 20190029027](https://ntrs.nasa.gov/citations/20190029027) (Cowan,
  Bond, Metcalf 2019) and
  [NTRS 20220003097](https://ntrs.nasa.gov/citations/20220003097) (Cowan
  et al. 2022) — the actual ISS Port 1 EATCS ammonia leak reports.

**Simulated**

- The lunar habitat itself. We do not claim to operate a lunar base. Real
  EDEN ISS telemetry is replayed *as if it were* coming from a lunar habitat
  life-support system.
- The lunar context (clock, day/night, comm windows). Visualization layers;
  the underlying data is real.
- One injected anomaly signature on top of real data: a slow coolant-leak
  pattern adapted from the ISS P1 EATCS reports. We replicate the
  *signature shape* (slow pressure decay → controller-lagged temperature
  drift → stepwise valve compensation) on EDEN ISS thermal control sensors.
  Magnitudes are tuned for a greenhouse-scale loop. We do **not** claim the
  demo is a literal ammonia leak — EDEN ISS does not measure ammonia.

**What we do not claim**

- We do not claim to "solve in seconds what NASA took years to solve."
  NASA's challenge was hypothesis validation against partial telemetry,
  root-cause analysis under operational constraints, and engineering
  remediation via EVA. We demonstrate the *initial pattern recognition*
  layer and the structured-evidence trace that ground controllers would
  expect from a tier-1 monitoring tool.

---

## Architecture

Seven layers behind interface boundaries — detectors, telemetry sources,
LLM backends, and scenarios are all swap-in components.

```
EDEN ISS data ──▶ ScenarioInjector ──▶ async telemetry bus
                                            │
                                            ├─▶ DAMP (matrix profile)
                                            ├─▶ ThresholdDetector
                                            ▼
                                       AnomalyEvent
                                            │
                                            ▼
                                  ReasoningAgent (Qwen 2.5 32B)
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                ▼              ▼              ▼              ▼          ▼
        query_sensor    fetch_subsystem  correlate_      lookup_      Diagnosis
         _history          _state         signals      failure_mode   (cited)
                                            │
                                            ▼
                                  Failure mode KB (NTRS-cited)
```

- **Layer 1 — Data**: `EdenIssReplayer` streams the dataset at configurable
  speed; `ScenarioInjector` wraps it transparently and applies registered
  anomaly modules. Both implement a single `TelemetrySource` protocol.
- **Layer 2 — Scenarios**: plug-in modules conforming to the
  `AnomalyModule` protocol. Registered via decorator at import time, loaded
  from YAML — no hard-coded scenario references in the base pipeline.
- **Layer 3 — Bus**: in-process asyncio queue with a rolling
  `TelemetryWindow`.
- **Layer 4 — Detection**: parallel `AnomalyDetector` implementations.
  DAMP runs the matrix profile on TCS pressure and temperature sensors;
  `ThresholdDetector` is a rule-based fallback. Detector outputs feed a
  bounded priority queue that dispatches investigations.
- **Layer 5 — Knowledge base**: typed `FailureMode` records (Pydantic v2)
  loaded from per-entry YAML, retrieved via structured symptom matching —
  no embedding similarity, no vector store. Five entries currently
  populated.
- **Layer 6 — Reasoning agent**: Qwen 2.5 32B Instruct served on AMD MI300X
  via vLLM/ROCm, called with OpenAI-compatible function-calling and a
  strict `Diagnosis` JSON output schema. Every tool call and result is
  emitted as a structured trace event for the frontend.
- **Layer 7 — API + frontend**: FastAPI WebSocket pipeline (`/telemetry`,
  `/agent_events`, `/scenario/start`, `/scenario/reset`) and a Gradio
  Blocks UI with three.js habitat scene, sparkline-equipped sensor table,
  and live investigation trace.

The same pipeline runs against any `TelemetrySource` and any
`AnomalyModule` — no code changes needed to point it at a different dataset
or scenario.

---

## Scenarios

The implemented headline scenario:

**`thermal_loop_coolant_leak`** — injected on EDEN ISS Thermal Control
System sensors. Slow pressure decay (sub-threshold for the first hour, then
accelerating) on `tcs/pressure-ams`; coupled inlet/outlet temperature drift
on `tcs/temp-ams_in`/`tcs/temp-ams_out` after a controller compensation lag;
stepwise valve opening on `tcs/valve-ams` as the loop controller responds to
falling pressure. Reproduces the *signature pattern* of the ISS P1 EATCS
ammonia leak — not its physics.

---

## Knowledge base

Five hand-curated entries with full source citations:

- **`iss_p1_eatcs_leak_2011`** — ISS Port 1 EATCS Ammonia Leak (NTRS
  20190029027, NTRS 20220003097)
- **`iss_cdra_bed_saturation`** — CDRA bed saturation pattern
- **`eden_iss_pump_degradation`** — synthetic NDS pump/circulation archetype
  (Rewicki et al. 2024)
- **`eden_iss_co2_scrubber_drift`** — synthetic AMS CO₂ baseline drift
  (Rewicki et al. 2024)
- **`eden_iss_illumination_degradation`** — synthetic ICS PAR degradation
  archetype (Rewicki et al. 2024)

Each entry has `primary_signature`, `secondary_signature`, `typical_onset`,
`distinguishing_features`, `differential_diagnosis`, `historical_context`,
`citations`, and `typical_response`. Entries that are hypothesized
archetypes rather than documented incidents say so explicitly in the
`historical_context` field.

---

## Running locally

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) for the backend
- The EDEN ISS dataset under `selene/data/eden_iss/edeniss2020/`
  (gitignored — download from
  [Zenodo](https://zenodo.org/records/11485183))
- A vLLM endpoint serving Qwen 2.5 32B Instruct (or a 14B fallback). Set
  `SELENE_LLM_BASE_URL`, `SELENE_LLM_API_KEY`, `SELENE_LLM_MODEL`.

### Backend

```bash
cd selene
poetry install
poetry run selene-serve --port 8001 --data-path data/eden_iss/edeniss2020
```

### Live frontend

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

BACKEND_URL=http://localhost:8001 \
BACKEND_WS_URL=ws://localhost:8001 \
python app.py
```

Open <http://127.0.0.1:7860>, pick `thermal_loop_coolant_leak`, click
**▶ Start**.

### Offline demo (no backend, no LLM)

For Hugging Face Spaces submissions or a network-isolated demo, replay a
captured run:

```bash
cd frontend
source .venv/bin/activate
python app_demo.py            # uses panels.py + style.css
# or, single-file variant:
python app_demo_inlined.py    # everything inlined except the recording JSON
```

Both read `frontend/demo_recordings/thermal_loop_coolant_leak.json` and
replay every captured telemetry frame and agent event at its original
cadence.

### Capturing a fresh recording

```bash
cd selene
poetry run python scripts/capture_demo_recordings.py \
    --backend http://localhost:8001 \
    --duration 90 \
    --output ../frontend/demo_recordings/thermal_loop_coolant_leak.json
```

The script also runs a validation report against the captured run
(pressure decay, temperature drift, valve stepping, DAMP firing,
investigation completion, KB-match correctness) and exits non-zero if any
required check fails.

---

## What this project is not

- Not a real lunar habitat simulator with physics
- Not a real-time control system
- Not a fine-tuned domain model — the LLM does synthesis and explanation
  over typed tool outputs and the KB, not fact generation
- Not a novel anomaly detection algorithm
- Not a novel agent framework

The contribution is the *integration*: detection → KB-grounded reasoning →
structured diagnosis with citations, end-to-end on a single AMD MI300X.

---

## References

- EDEN ISS 2020 Telemetry Dataset — <https://zenodo.org/records/11485183>
- Rewicki, Gawlikowski, Niebling, Denzler. *Unraveling Anomalies in Time:
  Unsupervised Discovery and Isolation of Anomalous Behavior in
  Bio-regenerative Life Support System Telemetry.* DLR, 2024.
  <https://arxiv.org/abs/2406.09825>
- NASA NTRS 20190029027 — *The International Space Station (ISS) Port 1 (P1)
  External Active Thermal Control System (EATCS) Ammonia Leak.* Cowan,
  Bond, Metcalf, 2019. <https://ntrs.nasa.gov/citations/20190029027>
- NASA NTRS 20220003097 — *Coolant Leak from ISS External Active Thermal
  Control System (EATCS) — An Examination of Most Probable.* Cowan et al.,
  2022. <https://ntrs.nasa.gov/citations/20220003097>
- vLLM ROCm —
  <https://docs.vllm.ai/en/latest/getting_started/amd-installation.html>
- DAMP / Matrix Profile — <https://stumpy.readthedocs.io>

---

## License

[MIT](LICENSE)
