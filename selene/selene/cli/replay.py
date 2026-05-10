"""CLI entry point: replay EDEN ISS telemetry with optional scenario injection.

Usage:
    python -m selene.cli.replay \\
        --data-path data/eden_iss/edeniss2020 \\
        --scenario scenarios/test_step_change.yaml \\
        --speed 60 \\
        [--start 2020-06-01T00:00:00Z] \\
        [--end 2020-06-07T23:59:59Z]

    # As fast as possible (tests / non-interactive):
    python -m selene.cli.replay --data-path ... --scenario ... --speed max
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string into an aware datetime (UTC if no offset given)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="selene-replay",
        description="Replay EDEN ISS telemetry, optionally with a scenario injected.",
    )
    p.add_argument("--data-path", required=True, type=Path,
                   help="Path to the EDEN ISS dataset root (contains edeniss2020.csv)")
    p.add_argument("--scenario", type=Path, default=None,
                   help="Path to a scenario YAML config file")
    p.add_argument("--speed", default="1.0",
                   help="Replay speed multiplier, or 'max' for as-fast-as-possible")
    p.add_argument("--start", type=_parse_iso, default=None,
                   help="Start timestamp (ISO-8601). Defaults to data start.")
    p.add_argument("--end", type=_parse_iso, default=None,
                   help="End timestamp (ISO-8601). Defaults to data end.")
    return p


async def _run(args: argparse.Namespace) -> None:
    from selene.data.injector import ScenarioInjector
    from selene.data.replayer import EdenIssReplayer
    from selene.scenarios.loader import load_scenario

    speed: float | None
    if args.speed == "max":
        speed = None
    else:
        speed = float(args.speed)

    replayer = EdenIssReplayer(
        data_path=args.data_path,
        start_time=args.start,
        end_time=args.end,
        speed_multiplier=speed,
    )

    meta = replayer.get_metadata()
    print(
        f"[selene-replay] data range: "
        f"{replayer._wide.index[0]} → {replayer._wide.index[-1]}",
        file=sys.stderr,
    )
    print(f"[selene-replay] sensors loaded: {len(meta.sensors)}", file=sys.stderr)

    source = replayer

    if args.scenario is not None:
        module = load_scenario(args.scenario)
        gt = module.get_ground_truth()
        print(f"[selene-replay] scenario: {module.name}", file=sys.stderr)
        print(f"[selene-replay] affected sensors: {module.affected_sensors}", file=sys.stderr)
        print(
            f"[selene-replay] ground truth window: {gt.start_time} → {gt.end_time}",
            file=sys.stderr,
        )
        source = ScenarioInjector(replayer, [module])  # type: ignore[arg-type]

    async for frame in source.stream():
        sys.stdout.write(frame.model_dump_json() + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
