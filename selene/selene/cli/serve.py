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
    return p


def main() -> None:
    import uvicorn

    args = _build_parser().parse_args()

    # Inject data path into the app module before starting so the lifespan
    # picks it up.
    import selene.api.app as api_app
    api_app._DATA_PATH = args.data_path

    uvicorn.run(
        "selene.api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
