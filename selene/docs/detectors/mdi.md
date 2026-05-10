# MDI Detector — Maximally Divergent Intervals

`selene/detection/mdi.py` — a multivariate anomaly detector that flags time intervals where the *joint* distribution of a chosen sensor group has drifted away from a baseline distribution. It complements DAMP (subsequence/discord-based, single-sensor) and Threshold (rule-based, single-sensor) by catching the case **no single sensor would flag**.

## When to reach for MDI

Use MDI when the anomaly you care about is correlation- or coupling-based rather than amplitude-based. Examples:

- Two sensors that are normally anti-correlated (a closed loop where pressure rises imply temperature drops) but decouple during a fault. Marginal means and variances may be unchanged.
- A loop whose mean operating point is preserved but whose variance/cross-covariance structure has shifted (e.g. control oscillation appears on two sensors at once).
- A subsystem where any single sensor could plausibly explore a wide range, so per-sensor thresholds are too loose, but the joint state is much more constrained.

If the anomaly shows up as a single sensor leaving its operational range, prefer Threshold. If it shows up as a single sensor exhibiting a discord-like subsequence, prefer DAMP.

## Algorithm

MDI compares two multivariate Gaussian fits over a sliding window:

1. **Baseline fit.** Take the first `baseline_window_size` samples of the telemetry window and fit a multivariate Gaussian `N(μ_b, Σ_b)` over the chosen sensors. This is done once.
2. **Sliding detection.** Slide a window of length `detection_window_size` across the remainder of the data. At each position fit `N(μ_d, Σ_d)` over the same sensors and compute the closed-form KL divergence between the two Gaussians:

   ```
   KL(detect ‖ baseline) = ½ · [ log(|Σ_b| / |Σ_d|)
                                  + tr(Σ_b⁻¹ Σ_d)
                                  + (μ_b − μ_d)ᵀ Σ_b⁻¹ (μ_b − μ_d)
                                  − k ]
   ```

   where `k` is the number of sensors. KL is non-negative analytically; tiny negative values from floating-point noise are clamped to zero.
3. **Interval grouping.** Mark every detection-window position where `KL > threshold` as anomalous. Group contiguous runs of anomalous positions into a single interval. Emit one `AnomalyEvent` per run, anchored at the timestamp of the run's peak score.

Each emitted event reports all selected sensors as `affected_sensors` because MDI's signal is a property of the joint distribution — there is no principled way to attribute the divergence to one specific sensor from the KL value alone. Use a single-sensor detector (DAMP, Threshold) on the same window if you need attribution.

## Why Gaussian fits?

The Gaussian assumption is a deliberate simplification:

- It admits a closed-form KL divergence (no Monte Carlo or histogram bin-edge sensitivity).
- It captures both first-order shifts (mean) and second-order shifts (covariance, including cross-correlation between sensors) in a single scalar.
- Subsystem telemetry is often locally near-Gaussian over windows of minutes-to-hours even when globally non-Gaussian.

It is wrong when the data is strongly multimodal, has heavy tails, or sits on a low-dimensional manifold within the sensor space. In those cases MDI's score is still informative as a *change* signal — what matters is the *difference* between baseline and detection fits, not the absolute fidelity of either fit — but the threshold may need empirical tuning rather than analytical interpretation.

## Numerical robustness

Two failure modes are handled in code:

- **Singular or near-singular covariance.** Two perfectly correlated sensors produce a rank-deficient Σ and `inv(Σ)` blows up. The code adds a Tikhonov ridge `ε·I` to every covariance matrix, with `ε` scaled by the mean diagonal entry so the regularization stays meaningful regardless of sensor units. The constant is `_COV_RIDGE = 1e-6` in the source.
- **Frames with missing or NaN readings.** Such frames are dropped from the matrix entirely, in line with `docs/eden_iss_format.md`'s "skip NaN, do not interpolate" preprocessing rule. Interpolation across gaps would create artificial correlations and contaminate both fits.

If, after dropping NaN/missing frames, the remaining sample count is below `baseline_window_size + detection_window_size`, the detector logs a debug message and returns `[]` rather than emitting partial fits.

## Parameters and tuning

| Parameter | Effect | Practical guidance |
|---|---|---|
| `sensor_ids` | The sensors that participate in the joint distribution. | Pick a coherent group — sensors on the same physical loop, not a random union. Adding unrelated sensors dilutes the signal. |
| `baseline_window_size` | Number of samples used to fit `N(μ_b, Σ_b)`. | Large enough that the baseline covariance is stable (≥ 50–200 samples for ~5 sensors). Small enough that it doesn't span an obvious operational regime change. |
| `detection_window_size` | Number of samples in each sliding fit. | Roughly the timescale of the anomaly you want to catch. Too short → noisy KL. Too long → slow to react and dilutes a brief event. |
| `threshold` | KL value above which a detection window is flagged. | Empirical. Run the detector on a known-clean window and pick a threshold above the observed quiescent KL. KL units are dimensionless but scale with `k` (sensor count) and the magnitude of the shift. |

## Event payload

Each emitted `AnomalyEvent` carries:

- `detector_name = "mdi"`
- `timestamp` — the timestamp of the peak-KL detection window in the run.
- `affected_sensors` — every sensor in `sensor_ids` (joint signal, see above).
- `score` — the peak KL value within the run.
- `details`:
  - `interval_start`, `interval_end` — ISO-8601 boundaries of the contiguous above-threshold run, covering the full span of the detection windows in the run.
  - `baseline_window_size`, `detection_window_size` — echoes of the configured window sizes, useful when several MDI instances run with different settings.
  - `n_above_threshold` — how many sliding positions in this run exceeded the threshold (a coarse proxy for how sustained the anomaly was).
  - `kl_peak` — same as `score`, kept for symmetry with detail-only consumers.

## Worked example

A two-sensor stream where the marginals stay `N(0, 1)` but the joint covariance changes mid-stream:

```python
import numpy as np
from selene.detection.mdi import MdiDetector

rng = np.random.default_rng(42)
x_base = rng.standard_normal(200)
y_base = -x_base + 0.05 * rng.standard_normal(200)   # anti-correlated baseline

x_anom = rng.standard_normal(100)
y_anom = rng.standard_normal(100)                    # decoupled (independent)

x = np.concatenate([x_base, x_anom])
y = np.concatenate([y_base, y_anom])
# build a TelemetryWindow from x and y here ...

detector = MdiDetector(
    sensor_ids=["x", "y"],
    baseline_window_size=200,
    detection_window_size=40,
    threshold=2.0,
)
events = await detector.evaluate(window)
```

Both `x` and `y` keep mean ≈ 0 and std ≈ 1 across baseline and anomaly, so a per-sensor threshold detector cannot trigger. MDI flags the decoupling because `Σ_b` is concentrated along the line `y = −x` while `Σ_d` is roughly isotropic — the `log|Σ_b| / |Σ_d|` and `tr(Σ_b⁻¹ Σ_d)` terms grow large together. This case is exercised in `tests/detection/test_mdi.py::TestMdiDetectorMultivariate`.

## Known limitations

- **Static baseline.** The baseline window is fixed at the start of the input. If a real deployment expects a slowly drifting baseline, wrap the detector in a rolling-baseline scheduler at the pipeline level rather than rebuilding the detector internally.
- **No attribution.** MDI tells you *that* the joint distribution shifted, not *which* sensor is responsible. Pair with DAMP or Threshold on the same sensor group when attribution matters.
- **Gaussian assumption.** See "Why Gaussian fits?" above.
- **Sensitive to baseline window choice.** A baseline that accidentally includes part of an anomaly will undercount the anomaly. A baseline that is too short will produce an unstable `Σ_b` whose inverse amplifies noise. The sample-count guidance in the parameter table is the safer default.

## References

- Barz, Guanche Garcia, Rodner, Denzler — *Detecting Regions of Maximal Divergence for Spatio-Temporal Anomaly Detection.* IEEE TPAMI, 2018. (Original MDI formulation.)
- Rewicki, Gawlikowski, Niebling, Denzler — *Unraveling Anomalies in Time: Unsupervised Discovery and Isolation of Anomalous Behavior in Bio-regenerative Life Support System Telemetry.* arXiv:2406.09825, 2024. (MDI applied to EDEN ISS data; the dataset and benchmark this project builds on.)
