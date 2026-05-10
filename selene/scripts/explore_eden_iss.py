"""Exploration script for the EDEN ISS 2020 dataset (Step 1.2 of PLAN.md).

Loads one week of the dataset and prints:
  - subsystems found, total sensor count
  - sample timestamp range and sampling rate
  - TCS sensor units
  - a CSV head for one TCS sensor
  - pressure / temperature / flow-related sensor IDs in the TCS

Usage:
    poetry run python scripts/explore_eden_iss.py [--data-root data/eden_iss/edeniss2020]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "eden_iss" / "edeniss2020"

# One week, chosen to be inside the dataset and away from year boundaries.
WEEK_START = "2020-06-01 00:00:00"
WEEK_END = "2020-06-08 00:00:00"


def load_sensor_index(data_root: Path) -> pd.DataFrame:
    """Read the dataset-level metadata CSV (filename, subsystem, type, unit)."""
    index_path = data_root / "edeniss2020.csv"
    if not index_path.exists():
        sys.exit(f"Missing metadata file: {index_path}")
    df = pd.read_csv(index_path)
    df["sensor_id"] = df["Filename"].str.removesuffix(".csv")
    df["full_path"] = df["Path"].apply(lambda p: data_root / p)
    return df


def load_one_week(sensor_path: Path, start: str, end: str) -> pd.DataFrame:
    """Load one CSV and filter to the [start, end) window."""
    df = pd.read_csv(sensor_path, parse_dates=["time"])
    mask = (df["time"] >= start) & (df["time"] < end)
    return df.loc[mask].reset_index(drop=True)


def infer_sampling_rate(timestamps: pd.Series) -> float:
    """Return the median spacing between consecutive samples, in seconds."""
    deltas = timestamps.diff().dropna().dt.total_seconds()
    return float(deltas.median())


def categorize_tcs_sensors(tcs_index: pd.DataFrame) -> dict[str, list[str]]:
    """Bucket TCS sensors by physical category, using metadata + name patterns."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for _, row in tcs_index.iterrows():
        sid = row["sensor_id"]
        short_type = row["Sensor Type (short)"]
        if short_type == "P":
            buckets["pressure"].append(sid)
        elif short_type == "T":
            buckets["temperature"].append(sid)
        elif short_type == "VALVE":
            # Valve openings are the closest proxy for flow rate in this dataset:
            # they modulate coolant flow through TCS branches. There are no
            # dedicated flow sensors in the EDEN ISS TCS subsystem.
            buckets["flow_proxy_valve"].append(sid)
        elif short_type == "RH":
            buckets["humidity"].append(sid)
        else:
            buckets[f"other_{short_type}"].append(sid)
    return dict(buckets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--week-start", default=WEEK_START)
    parser.add_argument("--week-end", default=WEEK_END)
    args = parser.parse_args()

    if not args.data_root.exists():
        sys.exit(f"Data root not found: {args.data_root}")

    print(f"# EDEN ISS exploration — {args.week_start} to {args.week_end}")
    print(f"# Data root: {args.data_root}")
    print()

    index = load_sensor_index(args.data_root)

    # 1. Subsystem list and sensor count
    subsystem_counts = index.groupby("Subsystem").size().sort_index()
    print("## Subsystems and sensor counts")
    for subsystem, count in subsystem_counts.items():
        print(f"  {subsystem:10s} {count:3d} sensors")
    print(f"  {'TOTAL':10s} {len(index):3d} sensors")
    print()

    # 2. Sample timestamp range + sampling rate (use one TCS sensor)
    sample_path = (index[index["Subsystem"] == "TCS"].iloc[0])["full_path"]
    sample = load_one_week(sample_path, args.week_start, args.week_end)
    if sample.empty:
        sys.exit(f"No rows in week window for {sample_path}. Pick a different week.")
    rate_seconds = infer_sampling_rate(sample["time"])
    print("## Timestamp range and sampling rate (from one TCS sensor)")
    print(f"  source_file:   {sample_path.relative_to(args.data_root)}")
    print(f"  first sample:  {sample['time'].iloc[0]}")
    print(f"  last sample:   {sample['time'].iloc[-1]}")
    print(f"  row count:     {len(sample)}")
    print(f"  median delta:  {rate_seconds:.1f} seconds")
    print(f"  sampling rate: {1.0 / rate_seconds:.6f} Hz")
    print()

    # 3. TCS units
    tcs_index = index[index["Subsystem"] == "TCS"].copy()
    print("## TCS sensors — units")
    for _, row in tcs_index.iterrows():
        print(f"  {row['sensor_id']:25s} {row['Sensor Type (long)']:35s} unit={row['Unit']!r}")
    print()

    # 4. CSV head for the TCS subsystem (use the first pressure sensor as the representative example)
    head_target = (tcs_index[tcs_index["Sensor Type (short)"] == "P"].iloc[0])["full_path"]
    print(f"## CSV head — {head_target.relative_to(args.data_root)}")
    with head_target.open() as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            print(f"  {line.rstrip()}")
    print()

    # 5. TCS sensor IDs by category
    print("## TCS sensors by physical category")
    buckets = categorize_tcs_sensors(tcs_index)
    for category, sensor_ids in sorted(buckets.items()):
        print(f"  {category} ({len(sensor_ids)}):")
        for sid in sensor_ids:
            print(f"    - {sid}")
    print()

    # 6. Missing-value scan across the chosen week (lightweight: count NaNs per file)
    print("## Missing-value scan across one week")
    issues = 0
    for _, row in index.iterrows():
        df = load_one_week(row["full_path"], args.week_start, args.week_end)
        nan_count = int(df.iloc[:, 1].isna().sum())
        if nan_count > 0:
            print(f"  {row['sensor_id']:30s} NaN rows: {nan_count}")
            issues += 1
    if issues == 0:
        print("  No NaN values found in the chosen week across all 97 sensors.")
    print()

    # 7. Verify uniformity: do all sensors share the same row count in this window?
    print("## Row-count uniformity across the week")
    row_counts: dict[int, int] = defaultdict(int)
    for _, row in index.iterrows():
        df = load_one_week(row["full_path"], args.week_start, args.week_end)
        row_counts[len(df)] += 1
    for n_rows, n_files in sorted(row_counts.items()):
        print(f"  {n_files:3d} files have {n_rows} rows")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
