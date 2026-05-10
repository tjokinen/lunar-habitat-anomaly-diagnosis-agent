# Selene

A lunar habitat anomaly diagnosis agent that integrates EDEN ISS data for intelligent fault detection and reasoning.

## Overview

Selene is an AI-powered diagnostic system designed to:
- Ingest and replay EDEN ISS telemetry data
- Detect anomalies using multiple detection methods (DAMP, MDI, threshold-based)
- Reason about failures using a knowledge base of failure modes
- Provide interactive diagnosis and remediation recommendations via CLI and API

## Architecture

- **core**: Interfaces and shared types
- **data**: EDEN ISS replayer and telemetry ingestion
- **scenarios**: Anomaly scenarios and YAML-based configuration
- **detection**: Anomaly detection algorithms (DAMP, MDI, threshold detectors)
- **knowledge**: Failure mode knowledge base
- **agent**: Reasoning agent and diagnostic tools
- **api**: FastAPI REST and WebSocket endpoints
- **cli**: Command-line interface for local diagnosis

## Data

Telemetry comes from the EDEN ISS 2020 dataset (Zenodo record `10.5281/zenodo.11485183`). See [`docs/eden_iss_format.md`](docs/eden_iss_format.md) for the format reference, sensor inventory, sampling cadence, and preprocessing notes that the replayer relies on.

## Setup

```bash
cd selene
poetry install
```

## Development

Run tests:
```bash
poetry run pytest
```

Lint and type-check:
```bash
poetry run ruff check .
poetry run mypy .
```

## License

MIT
