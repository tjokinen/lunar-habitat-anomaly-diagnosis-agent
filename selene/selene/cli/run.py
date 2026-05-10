"""CLI entry point: run the full Selene pipeline end-to-end.

Usage:
    python -m selene.cli.run \\
        --data-path data/eden_iss/edeniss2020 \\
        --scenario scenarios/thermal_loop_coolant_leak.yaml \\
        --speed 60

Each pipeline event (TelemetryFrame, AnomalyEvent, AgentEvent) is written
to stdout as a JSON object (one per line).  Diagnostic metadata goes to
stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="selene-run",
        description="Run the full Selene pipeline (telemetry → detectors → agent).",
    )
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--scenario", type=Path, default=None)
    p.add_argument(
        "--speed",
        default="60.0",
        help="Replay speed multiplier or 'max' for as-fast-as-possible",
    )
    p.add_argument("--start", type=_parse_iso, default=None)
    p.add_argument("--end", type=_parse_iso, default=None)
    p.add_argument(
        "--window",
        type=float,
        default=3600.0,
        help="Rolling detector window in seconds (default 3600)",
    )
    return p


async def _run(args: argparse.Namespace) -> None:
    from selene.agent.agent import ReasoningAgent
    from selene.agent.store import TelemetryStore
    from selene.agent.types import Event
    from selene.core.interfaces import AnomalyEvent, TelemetryFrame
    from selene.data.injector import ScenarioInjector
    from selene.data.replayer import EdenIssReplayer
    from selene.detection.damp import DampDetector
    from selene.detection.threshold import ThresholdDetector
    from selene.knowledge.loader import load_kb
    from selene.pipeline import run_pipeline
    from selene.scenarios.loader import load_scenario

    speed: float | None = None if args.speed == "max" else float(args.speed)

    replayer = EdenIssReplayer(
        data_path=args.data_path,
        start_time=args.start,
        end_time=args.end,
        speed_multiplier=speed,
    )
    meta = replayer.get_metadata()

    print(
        f"[selene-run] data range: "
        f"{replayer._wide.index[0]} → {replayer._wide.index[-1]}",
        file=sys.stderr,
    )
    print(f"[selene-run] sensors loaded: {len(meta.sensors)}", file=sys.stderr)

    source = replayer
    if args.scenario is not None:
        module = load_scenario(args.scenario)
        gt = module.get_ground_truth()
        print(f"[selene-run] scenario: {module.name}", file=sys.stderr)
        print(f"[selene-run] affected sensors: {module.affected_sensors}", file=sys.stderr)
        print(
            f"[selene-run] ground truth: {gt.start_time} → {gt.end_time}",
            file=sys.stderr,
        )
        source = ScenarioInjector(replayer, [module])  # type: ignore[arg-type]

    # Build detectors
    config_path = Path(__file__).parent.parent.parent / "config" / "sensor_ranges.yaml"
    detectors = []
    if config_path.exists():
        detectors.append(ThresholdDetector.from_yaml(config_path, metadata=meta))
    tcs_sensors = [
        sid for sid, info in meta.sensors.items()
        if info.get("subsystem") == "TCS" and info.get("sensor_type") in ("P", "T")
    ]
    if tcs_sensors:
        detectors.append(
            DampDetector(sensor_ids=tcs_sensors[:6], window_length=12, threshold=2.0)
        )
    print(f"[selene-run] detectors: {[d.name for d in detectors]}", file=sys.stderr)

    kb = load_kb()
    store = TelemetryStore(retention=timedelta(hours=2))
    agent = ReasoningAgent(store=store, kb=kb, metadata=meta)

    async def event_handler(event: Event) -> None:
        if isinstance(event, TelemetryFrame):
            sys.stdout.write(event.model_dump_json() + "\n")
        elif isinstance(event, AnomalyEvent):
            sys.stdout.write(event.model_dump_json() + "\n")
        else:
            sys.stdout.write(event.model_dump_json() + "\n")  # type: ignore[union-attr]
        sys.stdout.flush()

    await run_pipeline(
        telemetry_source=source,  # type: ignore[arg-type]
        detectors=detectors,
        agent=agent,
        event_handler=event_handler,
        window_size=timedelta(seconds=args.window),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
