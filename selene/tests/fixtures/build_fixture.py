"""Script to regenerate the EDEN ISS fixture used in tests.

Run once from the selene/ package root:
    python tests/fixtures/build_fixture.py
"""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).parent / "eden_iss_sample"

TIMESTAMPS = [
    "2020-06-01 00:05:00",
    "2020-06-01 00:10:00",
    "2020-06-01 00:15:00",
    "2020-06-01 00:20:00",
    "2020-06-01 00:25:00",
]

SENSORS = {
    "tcs/pressure-ams":  ("bar",          "TCS",     "P",     [1.0, 1.01, 1.02, 1.03, 1.04]),
    "tcs/temp-ams_in":   ("degrees celsius", "TCS",  "T",     [20.0, 20.1, 20.2, 20.3, 20.4]),
    "tcs/valve-ams":     ("percent",       "TCS",    "VALVE", [50.0, 50.0, 51.0, 51.0, 52.0]),
    "nds/flow-rate-01":  ("litre",         "NDS",    "V",     [5.0, 5.1, 5.0, 4.9, 5.0]),
}


def build():
    ROOT.mkdir(parents=True, exist_ok=True)

    # Write edeniss2020.csv index
    with open(ROOT / "edeniss2020.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filename", "Path", "Subsystem", "Sensor Type (short)", "Sensor Type (long)", "Unit"])
        for sensor_id, (unit, subsystem, short, _) in SENSORS.items():
            filename = sensor_id.split("/")[-1] + ".csv"
            path = sensor_id + ".csv"
            w.writerow([filename, path, subsystem, short, short, unit])

    # Write per-sensor CSVs
    for sensor_id, (unit, subsystem, short, values) in SENSORS.items():
        subdir = ROOT / sensor_id.rsplit("/", 1)[0]
        subdir.mkdir(parents=True, exist_ok=True)
        csv_path = ROOT / f"{sensor_id}.csv"
        basename = sensor_id.split("/")[-1]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", basename])
            for ts, val in zip(TIMESTAMPS, values):
                w.writerow([ts, val])


if __name__ == "__main__":
    build()
    print(f"Fixture written to {ROOT}")
