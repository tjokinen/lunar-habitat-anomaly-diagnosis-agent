"""Capture a complete scenario run from the live backend for offline demo replay.

Subscribes to the backend's ``/telemetry`` and ``/agent_events`` WebSockets,
triggers the named scenario via ``POST /scenario/start``, and writes a single
JSON file containing every captured event with its wall-clock offset (in
seconds) from scenario start. The frontend's Demo Mode replays the file by
sleeping ``event.t - prev.t`` between dispatches, which preserves the original
look without depending on a live backend.

Two modes:

  Capture + analyze (default):
      poetry run python scripts/capture_demo_recordings.py \\
          --scenario thermal_loop_coolant_leak \\
          --backend http://localhost:8000 \\
          --duration 90 \\
          --output ../frontend/demo_recordings/thermal_loop_coolant_leak.json

  Analyze an existing recording (no backend required):
      poetry run python scripts/capture_demo_recordings.py \\
          --analyze-only ../frontend/demo_recordings/thermal_loop_coolant_leak.json

The analyzer validates the captured run against the expected scenario
behaviour (pressure decay, temperature drift after lag, valve stepping, DAMP
detection, investigation completion, KB match) and exits non-zero if any
required check fails so this can wire into CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import websockets

logger = logging.getLogger("capture_demo_recordings")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def capture(
    scenario_id: str,
    backend_url: str,
    duration_seconds: float,
    output_path: Path,
) -> dict:
    """Trigger *scenario_id* on the backend at *backend_url* and capture every
    WebSocket event for *duration_seconds* of wall time. Writes a recording
    JSON to *output_path* and returns the in-memory recording dict.
    """
    backend_url = backend_url.rstrip("/")
    ws_base = backend_url.replace("http://", "ws://").replace("https://", "wss://")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Sanity: backend reachable, scenario known.
        r = await client.get(f"{backend_url}/scenarios")
        r.raise_for_status()
        scenarios = {s["scenario_id"] for s in r.json()}
        if scenario_id not in scenarios:
            raise SystemExit(
                f"Unknown scenario {scenario_id!r}. Known: {sorted(scenarios)}"
            )

        # Reset backend so we capture from a clean baseline.
        await client.post(f"{backend_url}/scenario/reset")

        try:
            speed_resp = await client.get(f"{backend_url}/replay/speed")
            speed_multiplier = float(speed_resp.json().get("multiplier") or 60.0)
        except Exception:
            speed_multiplier = 60.0

    events: list[dict] = []
    stop = asyncio.Event()
    start_wall: float | None = None

    async def consume(channel: str, ws_url: str) -> None:
        try:
            async with websockets.connect(ws_url, max_size=2**24) as ws:
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        return
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        logger.warning("non-JSON message on %s: %r", channel, msg[:80])
                        continue
                    if start_wall is None:
                        # Drop pre-start frames (the reset above shouldn't emit any
                        # but the WS server may still flush a buffered telemetry
                        # frame from the previous nominal pipeline).
                        continue
                    t = time.monotonic() - start_wall
                    events.append(
                        {"t": round(t, 3), "channel": channel, "data": data}
                    )
        except Exception as exc:
            logger.warning("WS consumer for %s exited: %s", channel, exc)

    telemetry_task = asyncio.create_task(
        consume("telemetry", f"{ws_base}/telemetry")
    )
    agent_task = asyncio.create_task(
        consume("agent_events", f"{ws_base}/agent_events")
    )

    # Give the WS consumers a moment to connect before triggering the scenario.
    await asyncio.sleep(0.5)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Mark t=0 *immediately before* the start so the first telemetry frame
        # of the new pipeline shows up at a small positive t.
        start_wall = time.monotonic()
        r = await client.post(
            f"{backend_url}/scenario/start", json={"scenario_id": scenario_id}
        )
        r.raise_for_status()
        ground_truth = r.json().get("ground_truth")
        logger.info(
            "scenario %s started; capturing for %.1fs (speed=%.0fx)",
            scenario_id, duration_seconds, speed_multiplier,
        )

    await asyncio.sleep(duration_seconds)
    stop.set()

    # Drain.
    await asyncio.gather(telemetry_task, agent_task, return_exceptions=True)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(f"{backend_url}/scenario/reset")
        except Exception:
            pass

    events.sort(key=lambda e: e["t"])

    recording = {
        "metadata": {
            "scenario_id": scenario_id,
            "speed_multiplier": speed_multiplier,
            "captured_at_wall": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_seconds,
            "ground_truth": ground_truth,
        },
        "events": events,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(recording, f, indent=2)

    n_tel = sum(1 for e in events if e["channel"] == "telemetry")
    n_age = sum(1 for e in events if e["channel"] == "agent_events")
    logger.info(
        "wrote %d events (telemetry=%d, agent_events=%d) to %s",
        len(events), n_tel, n_age, output_path,
    )
    return recording


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


# Color codes for the report output.  Disabled when stdout isn't a tty.
_USE_COLOR = sys.stdout.isatty()
_GREEN = "\033[32m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_DIM = "\033[2m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def _pass(msg: str) -> None:
    print(f"  {_GREEN}PASS{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}FAIL{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}WARN{_RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _sensor_series(events: list[dict], sensor_id: str) -> list[tuple[datetime, float]]:
    """Extract (sim_timestamp, value) pairs for one sensor from telemetry events."""
    series: list[tuple[datetime, float]] = []
    for e in events:
        if e["channel"] != "telemetry":
            continue
        readings = e["data"].get("readings", {})
        r = readings.get(sensor_id)
        if r is None:
            continue
        try:
            ts = _parse_iso(r["timestamp"])
            v = float(r["value"])
        except (KeyError, ValueError, TypeError):
            continue
        series.append((ts, v))
    return series


def _split_by_window(
    series: list[tuple[datetime, float]],
    onset: datetime,
    end: datetime,
) -> tuple[list[float], list[float], list[float]]:
    """Split a sensor series into (pre_onset, during, post_anomaly) value lists."""
    pre, during, post = [], [], []
    for ts, v in series:
        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
        on_naive = onset.replace(tzinfo=None) if onset.tzinfo else onset
        en_naive = end.replace(tzinfo=None) if end.tzinfo else end
        if ts_naive < on_naive:
            pre.append(v)
        elif ts_naive <= en_naive:
            during.append(v)
        else:
            post.append(v)
    return pre, during, post


def analyze(recording: dict) -> int:
    """Run the validation suite over a recording. Returns 0 if all required
    checks pass, 1 if any required check fails."""
    meta = recording.get("metadata") or {}
    events = recording.get("events") or []
    gt = meta.get("ground_truth") or {}

    print()
    print("=" * 72)
    print(" ANALYSIS REPORT")
    print("=" * 72)
    print(f"Scenario:    {meta.get('scenario_id')}")
    print(f"Speed:       {meta.get('speed_multiplier')}x")
    print(f"Duration:    {meta.get('duration_seconds')}s wall")
    print(f"Captured:    {meta.get('captured_at_wall')}")

    if not events:
        _fail("no events captured — recording is empty")
        return 1

    n_tel = sum(1 for e in events if e["channel"] == "telemetry")
    n_age = sum(1 for e in events if e["channel"] == "agent_events")
    print(f"Events:      {len(events)} total  ({n_tel} telemetry, {n_age} agent)")

    if not gt or not gt.get("start_time") or not gt.get("end_time"):
        _fail("recording is missing ground_truth.start_time / end_time")
        return 1

    onset = _parse_iso(gt["start_time"])
    end = _parse_iso(gt["end_time"])
    print(f"Anomaly:     {onset.isoformat()} → {end.isoformat()}")
    print(f"Sensors:     {gt.get('affected_sensors')}")

    failures = 0

    # ----------------------------------------------------------------------
    _section("1. Coverage")
    sim_timestamps = []
    for e in events:
        if e["channel"] != "telemetry":
            continue
        ts_str = e["data"].get("timestamp")
        if ts_str:
            sim_timestamps.append(_parse_iso(ts_str))
    if not sim_timestamps:
        _fail("no telemetry frames had a parseable timestamp")
        failures += 1
    else:
        first_sim = min(sim_timestamps).replace(tzinfo=None)
        last_sim = max(sim_timestamps).replace(tzinfo=None)
        on_n = onset.replace(tzinfo=None)
        en_n = end.replace(tzinfo=None)
        print(f"  Sim time range: {first_sim} → {last_sim}")
        if first_sim >= on_n:
            _fail(f"no pre-onset frames captured (first sim frame {first_sim} ≥ onset {on_n})")
            failures += 1
        else:
            _pass(f"pre-onset frames captured (warmup spans {(on_n - first_sim).total_seconds()/60:.0f} min)")
        if last_sim < en_n:
            _warn(
                f"recording ended before anomaly window closed "
                f"(last sim {last_sim} < end {en_n}); some late checks may be partial"
            )
        elif last_sim < en_n.replace(year=en_n.year):
            pass
        if any(on_n <= ts.replace(tzinfo=None) <= en_n for ts in sim_timestamps):
            _pass("anomaly-window frames captured")
        else:
            _fail("no telemetry frames fall inside the anomaly window")
            failures += 1

    # ----------------------------------------------------------------------
    _section("2. Pressure decay (tcs/pressure-ams)")
    p_series = _sensor_series(events, "tcs/pressure-ams")
    if not p_series:
        _fail("no readings for tcs/pressure-ams")
        failures += 1
    else:
        pre, during, post = _split_by_window(p_series, onset, end)
        print(f"  Frames: pre={len(pre)} during={len(during)} post={len(post)}")
        if pre:
            print(f"  Pre-onset min/max: {min(pre):.4f} / {max(pre):.4f} bar")
        if during:
            print(f"  During    min/max: {min(during):.4f} / {max(during):.4f} bar")
            if pre:
                pre_max = max(pre)
                during_min = min(during)
                drop_pct = (pre_max - during_min) / pre_max * 100 if pre_max else 0
                print(f"  Max drop from pre-onset baseline: {drop_pct:.2f}%")
                if during_min < pre_max - 1e-6:
                    _pass(
                        f"pressure dropped during anomaly window "
                        f"({pre_max:.4f} → {during_min:.4f} bar)"
                    )
                else:
                    _fail("pressure did NOT decrease during the anomaly window")
                    failures += 1
                # Expect 6% max drop with default scenario params; tolerate >0.5%.
                if drop_pct < 0.5:
                    _warn(
                        f"pressure drop ({drop_pct:.2f}%) is much smaller than the "
                        "expected ~6% — recording may have ended early"
                    )
        if post and during:
            post_max = max(post)
            during_min = min(during)
            if post_max > during_min:
                _pass(
                    f"post-anomaly pressure recovered above the in-window minimum "
                    f"({during_min:.4f} → {post_max:.4f} bar)"
                )
            else:
                _warn("post-anomaly pressure did not recover above in-window minimum")

    # ----------------------------------------------------------------------
    _section("3. Temperature drift (tcs/temp-ams_in, tcs/temp-ams_out)")
    for tid in ("tcs/temp-ams_in", "tcs/temp-ams_out"):
        t_series = _sensor_series(events, tid)
        if not t_series:
            _warn(f"no readings for {tid}")
            continue
        pre, during, _ = _split_by_window(t_series, onset, end)
        # Only the post-30min portion of `during` should show injected rise;
        # report aggregate stats anyway.
        if pre and during:
            pre_avg = sum(pre) / len(pre)
            during_max = max(during)
            rise = during_max - pre_avg
            print(f"  {tid}: pre avg={pre_avg:.2f}°C, during max={during_max:.2f}°C, rise={rise:+.2f}°C")
            if rise > 0.1:
                _pass(f"{tid} rose by {rise:+.2f}°C above pre-onset average")
            else:
                _warn(f"{tid} did not show clear upward drift")
        elif during:
            print(f"  {tid}: during avg={sum(during)/len(during):.2f}°C (no pre-onset frames)")

    # ----------------------------------------------------------------------
    _section("4. Valve stepping (tcs/valve-ams)")
    v_series = _sensor_series(events, "tcs/valve-ams")
    if not v_series:
        _warn("no readings for tcs/valve-ams")
    else:
        pre, during, _ = _split_by_window(v_series, onset, end)
        if pre and during:
            pre_max = max(pre)
            during_max = max(during)
            jump = during_max - pre_max
            print(f"  Pre-onset max: {pre_max:.1f}%, during max: {during_max:.1f}%, jump: {jump:+.1f}pp")
            if jump >= 4.0:
                _pass(f"valve opened by {jump:+.1f}pp during anomaly window")
            elif jump > 0:
                _warn(f"valve opened only {jump:+.1f}pp — first step is +5pp at 1% loss")
            else:
                _fail("valve did not step up during anomaly window")
                failures += 1

    # ----------------------------------------------------------------------
    _section("5. DAMP detector firings")
    damp_events = [
        e for e in events
        if e["channel"] == "agent_events"
        and e["data"].get("detector_name") == "damp"
    ]
    print(f"  {len(damp_events)} DAMP AnomalyEvents in agent_events stream")
    if damp_events:
        scores = sorted({round(float(e["data"].get("score", 0)), 2) for e in damp_events})
        sensors = sorted({s for e in damp_events for s in e["data"].get("affected_sensors", [])})
        print(f"  Sensors flagged: {sensors}")
        print(f"  Score range: {min(scores):.2f} → {max(scores):.2f}")
        _pass("DAMP fired during the run")
    else:
        _fail("DAMP never fired — check window_size / threshold in the pipeline")
        failures += 1

    # ----------------------------------------------------------------------
    _section("6. Agent investigations")
    started = [
        e for e in events
        if e["channel"] == "agent_events"
        and e["data"].get("type") == "agent_run_started"
    ]
    completed = [
        e for e in events
        if e["channel"] == "agent_events"
        and e["data"].get("type") == "agent_run_completed"
    ]
    failed = [
        e for e in events
        if e["channel"] == "agent_events"
        and e["data"].get("type") == "agent_run_failed"
    ]
    print(f"  Started: {len(started)}  Completed: {len(completed)}  Failed: {len(failed)}")
    if started:
        _pass(f"{len(started)} investigation(s) triggered")
    else:
        _fail("no AgentRunStarted events — agent never investigated")
        failures += 1
        return failures

    if completed:
        _pass(f"{len(completed)} investigation(s) reached AgentRunCompleted")
    else:
        _fail("no investigations completed — check LLM output / max_iterations")
        failures += 1

    # ----------------------------------------------------------------------
    _section("7. Diagnosis quality (best completed run)")
    if not completed:
        _warn("skipped — no completed investigation")
    else:
        # Pick the highest-confidence completed diagnosis.
        best = max(
            completed,
            key=lambda e: float(e["data"].get("diagnosis", {}).get("confidence", 0)),
        )
        diag = best["data"].get("diagnosis", {}) or {}
        print(f"  Primary hypothesis: {diag.get('primary_hypothesis')}")
        print(f"  Confidence: {diag.get('confidence')}")
        print(f"  Matched failure modes: {diag.get('matched_failure_modes')}")
        n_cite = len(diag.get("citations") or [])
        print(f"  Citations: {n_cite}")
        n_evidence = len(diag.get("supporting_evidence") or [])
        n_actions = len(diag.get("recommended_actions") or [])
        print(f"  Evidence items: {n_evidence}, recommended actions: {n_actions}")

        conf = float(diag.get("confidence") or 0.0)
        if conf >= 0.4:
            _pass(f"confidence {conf:.2f} ≥ 0.40")
        else:
            _warn(f"confidence {conf:.2f} below 0.40")

        modes = diag.get("matched_failure_modes") or []
        if "iss_p1_eatcs_leak_2011" in modes:
            _pass("matched_failure_modes includes iss_p1_eatcs_leak_2011")
        else:
            _warn(
                "expected iss_p1_eatcs_leak_2011 in matched_failure_modes; "
                f"got {modes}"
            )

        if n_cite >= 1:
            _pass(f"{n_cite} citation(s) present")
        else:
            _warn("diagnosis has no citations")

        prim = (diag.get("primary_hypothesis") or "").lower()
        if any(k in prim for k in ("thermal", "leak", "coolant", "eatcs", "loop")):
            _pass("primary_hypothesis text mentions thermal/leak/coolant/eatcs/loop")
        else:
            _warn("primary_hypothesis text doesn't mention thermal/leak/coolant")

    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    if failures == 0:
        print(f" {_GREEN}OK{_RESET} — all required checks passed")
    else:
        print(f" {_RED}FAILED{_RESET} — {failures} required check(s) did not pass")
    print("=" * 72)
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Capture and/or analyse a scenario run for offline demo replay.",
    )
    p.add_argument("--scenario", default="thermal_loop_coolant_leak")
    p.add_argument("--backend", default="http://localhost:8000")
    p.add_argument(
        "--duration", type=float, default=90.0,
        help="Seconds of wall time to capture. At 600x speed, 90s = 15 sim hours, "
             "covering the 30-min warmup, 8-hour anomaly, and post-anomaly recovery.",
    )
    p.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[2]
            / "frontend" / "demo_recordings" / "thermal_loop_coolant_leak.json"
        ),
    )
    p.add_argument(
        "--analyze-only", metavar="PATH",
        help="Skip capture; load PATH and run analysis only.",
    )
    p.add_argument(
        "--no-analyze", action="store_true",
        help="Capture only; don't run analysis afterward.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.analyze_only:
        path = Path(args.analyze_only)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 2
        with path.open() as f:
            recording = json.load(f)
        return analyze(recording)

    output_path = Path(args.output)
    try:
        recording = asyncio.run(
            capture(args.scenario, args.backend, args.duration, output_path)
        )
    except httpx.ConnectError as exc:
        print(f"Cannot reach backend at {args.backend}: {exc}", file=sys.stderr)
        return 2
    except SystemExit:
        raise

    if args.no_analyze:
        return 0
    return analyze(recording)


if __name__ == "__main__":
    sys.exit(main())
