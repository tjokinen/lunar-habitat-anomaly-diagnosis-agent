"""CLI entry point: start the FastAPI backend."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="selene-serve",
        description="Start the Selene FastAPI backend.",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/eden_iss/edeniss2020"),
        help="Path to EDEN ISS dataset root",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="Replay speed multiplier (60 = 1 wall-sec per simulated minute; "
             "300 = 1 frame/sec at 5-min cadence; 600 = 2 frames/sec).",
    )
    return p


def main() -> None:
    import uvicorn

    args = _build_parser().parse_args()

    # Inject data path / speed into the app module before starting so the
    # lifespan picks them up.
    import selene.api.app as api_app
    api_app._DATA_PATH = args.data_path
    api_app._SPEED_MULTIPLIER = args.speed

    uvicorn.run(
        "selene.api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
