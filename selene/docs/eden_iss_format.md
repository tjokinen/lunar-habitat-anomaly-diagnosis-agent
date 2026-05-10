# EDEN ISS 2020 Telemetry — Data Format Reference

Reference notes for the dataset that backs Selene's telemetry layer. Source: Zenodo record [10.5281/zenodo.11485183](https://zenodo.org/records/11485183) (Rewicki, Norman, Vrakking; DLR; 2024).

## Layout

```
data/eden_iss/edeniss2020/
├── README.md             # upstream README from Zenodo
├── edeniss2020.csv       # dataset-level sensor index (filename, subsystem, unit)
├── ams-feg/*.csv         # 9 sensors  — AMS, Future Exploration Greenhouse
├── ams-ses/*.csv         # 8 sensors  — AMS, Service Section
├── ics/*.csv             # 38 sensors — Illumination Control System
├── nds/*.csv             # 24 sensors — Nutrient Delivery System
└── tcs/*.csv             # 18 sensors — Thermal Control System
```

97 sensors total. AMS is split into two physical sub-areas (FEG and SES), so for our purposes the dataset has **5 subsystems**. The replayer should expose all 5 distinctly; any collapsing of AMS-FEG + AMS-SES into a single "AMS" view happens at the visualization layer.

## File format

Every per-sensor CSV has exactly two columns and a header:

```
time,<sensor_id>
2020-01-01 00:05:00,1.3
2020-01-01 00:10:00,1.3
...
```

- `time` is a naive timestamp in `YYYY-MM-DD HH:MM:SS` (local time at the Antarctic site, no TZ offset). Treat as UTC for our pipeline — the absolute calendar value is irrelevant since we replay it on a synthetic clock.
- The value column header equals the **sensor ID** = filename without `.csv`. So `tcs/pressure-ams.csv` → sensor_id `pressure-ams`.
- All values are numeric (`float`), even valves and pH. Verified: zero NaN / empty rows across all 97 sensors during the inspected week.

## Sensor index file

`edeniss2020.csv` at the dataset root maps every CSV file to its subsystem and unit:

| Column                | Meaning                                               |
| --------------------- | ----------------------------------------------------- |
| `Filename`            | basename of the per-sensor CSV (e.g. `pressure-ams.csv`) |
| `Path`                | path relative to data root (e.g. `tcs/pressure-ams.csv`) |
| `Subsystem`           | one of `AMS-FEG`, `AMS-SES`, `ICS`, `NDS`, `TCS`      |
| `Sensor Type (short)` | one of `CO2`, `PAR`, `RH`, `T`, `VPD`, `EC`, `H`, `PH`, `P`, `V`, `VALVE` |
| `Sensor Type (long)`  | human-readable label (note: misspelled `Temperarture` in source) |
| `Unit`                | unit string; empty for unitless (pH)                  |

This is the canonical source of truth for sensor metadata — the replayer should build its `SensorMetadata` from this file rather than inferring from filenames.

## Sampling

- **Cadence:** uniform, 5-minute interval (300 s spacing, ≈3.33 mHz).
- **Coverage:** 2020-01-01 00:05:00 → 2020-12-30 23:55:00 (full third mission year).
- **Rows per sensor:** 105,119 across the year; **2,016 per week**; verified uniform across all 97 sensors in the inspected week (2020-06-01 to 2020-06-07).

There is no rate variation across files or across the year that we need to handle. The dataset is already aligned — every sensor has the same timestamp grid.

## Units summary by sensor type

| Short | Long                            | Unit              | Notes                                |
| ----- | ------------------------------- | ----------------- | ------------------------------------ |
| `CO2` | Carbon Dioxide                  | `ppm`             |                                      |
| `PAR` | Photosynthetically Active Rad.  | `umol/m^2/s`      |                                      |
| `RH`  | Relative Humidity               | `percent`         |                                      |
| `T`   | Temperature                     | `degrees celsius` | source spelling is `Temperarture`    |
| `VPD` | Vapor Pressure Deficit          | `mbar`            |                                      |
| `EC`  | Electrical Conductivity         | `mS/cm`           |                                      |
| `H`   | Level (tank)                    | `cm`              |                                      |
| `PH`  | pH Value                        | (none)            | unitless                             |
| `P`   | Pressure                        | `bar`             |                                      |
| `V`   | Volume (tank)                   | `litre`           |                                      |
| `VALVE` | Valve opening                 | `percent`         | 0 = closed, 100 = fully open         |

## TCS subsystem in detail

The TCS is the headline scenario's target. Inventory:

| Category    | Count | Sensor IDs                                                                                                   | Unit              |
| ----------- | ----- | ------------------------------------------------------------------------------------------------------------ | ----------------- |
| Pressure    | 3     | `pressure-ams`, `pressure-free`, `pressure-ics`                                                              | `bar`             |
| Temperature | 11    | `temp-ams_2`, `temp-ams_hs_in`, `temp-ams_in`, `temp-ams_out`, `temp-ext_in`, `temp-free_in`, `temp-free_out`, `temp-ics_in`, `temp-il_hs_in`, `temp-int_out`, `temp-iocs_out` | `degrees celsius` |
| Valve       | 3     | `valve-ams`, `valve-free`, `valve-ics`                                                                       | `percent`         |
| Humidity    | 1     | `rh-ams_2`                                                                                                   | `percent`         |

**Naming convention.** TCS sensors follow `<measurement>-<location>` where `location` corresponds to a thermal loop branch:

- `ams` — branch serving the Atmosphere Management System
- `free` — branch serving the Future Exploration Greenhouse
- `ics` — branch serving the Illumination Control System
- `ext`, `int`, `il`, `iocs` — external / internal / illumination / IOCS sub-points within the loop
- `_in` / `_out` suffixes — inlet / outlet of a heat exchanger
- `_hs_in` — heat-source inlet (likely the chiller side)
- `_2` — second instance at a location (e.g. redundant sensor)

This is enough structure to write the headline `thermal_loop_coolant_leak` scenario: pick one branch (e.g. `ams`), inject the leak signature on its `pressure-*` sensor, propagate compensating drift to the matched `temp-*_in`/`temp-*_out` pair, and step the matched `valve-*` opening upward as the controller compensates.

**Flow sensors.** *None.* There are no dedicated flow sensors in the TCS subsystem. The valve opening percentages are the closest available proxy for flow-rate behavior. Scenarios that want to express a "flow" change must do it via valve sensors. Document this in scenario YAMLs to avoid confusion.

## Loading patterns

**Single sensor:**

```python
import pandas as pd

df = pd.read_csv("data/eden_iss/edeniss2020/tcs/pressure-ams.csv", parse_dates=["time"])
```

**Whole subsystem as a wide multivariate frame:**

```python
import pandas as pd, glob
from functools import reduce

frames = [
    pd.read_csv(f, parse_dates=["time"])
    for f in sorted(glob.glob("data/eden_iss/edeniss2020/tcs/*.csv"))
]
wide = reduce(lambda a, b: a.merge(b, on="time"), frames).sort_values("time").reset_index(drop=True)
```

The merge is a no-op alignment because the timestamp grid is already shared across all sensors.

## Preprocessing implications for the replayer (Step 1.4)

- **No NaN handling required** for the Jun 1–7 2020 reference window. The replayer should still defensively skip frames where a value is NaN (so it works on any window), but the common path is clean. Decision: **skip NaN rows**, log a warning, do not interpolate. Interpolation would introduce fake correlations that detectors would learn from.
- **No rate normalization required.** The grid is uniform 5-min across all files and the year.
- **Memory budget.** One year of all 97 sensors as `float64` is ≈ 76 MB raw (97 × 105,119 × 8 B), well within RAM. Loading on demand by week is sufficient and aligns with the per-week scenario windows we'll use.
- **Time treatment.** Treat timestamps as opaque strings during ingest, parse to `datetime` once per file, then re-emit as ISO-8601 in `TelemetryFrame.timestamp`. The absolute year (2020) is irrelevant — the replayer's `speed_multiplier` controls real-world cadence, and the scenario YAML's `start_time` / `end_time` are interpreted relative to the data window the CLI selected.
- **Sensor IDs are already unique across the whole dataset** (the metadata's `Path` column disambiguates the two `co2-1.csv` files in `ams-feg/` and `ams-ses/`). Use `Path` (without `.csv`, with `/` as the namespace separator) as the canonical sensor ID — i.e. `tcs/pressure-ams`, not `pressure-ams`. This avoids collisions like `co2-1` appearing in two subsystems.

## Known upstream README quirks

For reference when reading the upstream `README.md`:
- TCS is described with "Temperature (T) | 12" but the metadata file lists 11. Trust the metadata.
- TCS is described with "Relative Humidity (RH) | 2" but only 1 (`rh-ams_2`) appears. Trust the metadata.
- Subsystem totals therefore add to 97 only when using metadata counts.
- "Temperarture" is misspelled in `Sensor Type (long)` for every T sensor. Don't fix it — match it exactly when joining against the metadata file.
