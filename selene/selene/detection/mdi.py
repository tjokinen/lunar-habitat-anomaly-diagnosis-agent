"""MDI detector — Maximally Divergent Intervals via KL divergence on Gaussian fits.

Reference: Barz et al. and Rewicki et al. 2024 (https://arxiv.org/abs/2406.09825).
This is a from-scratch numpy implementation of the Gaussian-KL flavor: fit a
multivariate Gaussian to a baseline window, slide a detection window across the
remainder, and flag intervals whose KL divergence from the baseline exceeds a
threshold.

The detector is multivariate-by-design: it shines when no single sensor moves
outside its normal range but the joint distribution shifts (e.g. two normally
correlated sensors decouple).
"""

from __future__ import annotations

import logging

import numpy as np

from selene.core.interfaces import AnomalyEvent, TelemetryWindow

logger = logging.getLogger(__name__)

# Tikhonov-style ridge added to covariance matrices to keep them invertible
# when sensors are perfectly correlated or a window has near-zero variance.
# Scaled by the mean diagonal entry to stay relevant across unit ranges.
_COV_RIDGE = 1e-6


class MdiDetector:
    """Maximally Divergent Intervals over a multivariate time series.

    Fits a multivariate Gaussian to a fixed baseline window at the start of the
    telemetry window, then slides a detection window across the remainder and
    computes KL(detect ‖ baseline) at each position. Contiguous runs of
    detection windows above ``threshold`` are emitted as a single AnomalyEvent
    per run, anchored at the timestamp of the run's peak.

    Args:
        sensor_ids: Sensors that participate in the joint distribution.
        baseline_window_size: Number of samples used to fit the baseline
            Gaussian. Taken from the start of the input window.
        detection_window_size: Number of samples in each sliding detection
            window over which the comparison Gaussian is fit.
        threshold: KL-divergence value above which a detection window is
            considered anomalous.
    """

    name = "mdi"

    def __init__(
        self,
        sensor_ids: list[str],
        baseline_window_size: int,
        detection_window_size: int,
        threshold: float,
    ) -> None:
        if baseline_window_size <= 1:
            raise ValueError("baseline_window_size must be > 1")
        if detection_window_size <= 1:
            raise ValueError("detection_window_size must be > 1")
        if not sensor_ids:
            raise ValueError("sensor_ids must be non-empty")
        self._sensor_ids = list(sensor_ids)
        self._baseline = baseline_window_size
        self._detect = detection_window_size
        self._threshold = threshold

    async def evaluate(self, window: TelemetryWindow) -> list[AnomalyEvent]:
        matrix, frame_indices = self._extract_matrix(window)
        if matrix is None:
            return []

        n_samples = matrix.shape[0]
        if n_samples < self._baseline + self._detect:
            logger.debug(
                "MdiDetector: %d usable samples < baseline (%d) + detection (%d) — skipping",
                n_samples,
                self._baseline,
                self._detect,
            )
            return []

        baseline = matrix[: self._baseline]
        mu_b, _, cov_b_inv, log_det_b = self._fit_gaussian(baseline)
        if cov_b_inv is None:
            logger.warning("MdiDetector: baseline covariance not invertible — skipping")
            return []

        scores: list[tuple[int, float]] = []  # (matrix_index_at_window_start, kl)
        last_start = n_samples - self._detect
        for start in range(self._baseline, last_start + 1):
            kl = self._kl_to_baseline(matrix[start : start + self._detect], mu_b, cov_b_inv, log_det_b)
            scores.append((start, kl))

        return self._scores_to_events(window, frame_indices, scores)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_matrix(
        self, window: TelemetryWindow
    ) -> tuple[np.ndarray | None, list[int]]:
        """Build (n_samples, n_sensors) matrix from frames; drop frames with any
        NaN or missing reading on the selected sensors."""
        n_sensors = len(self._sensor_ids)
        rows: list[list[float]] = []
        frame_indices: list[int] = []

        for frame_idx, frame in enumerate(window.frames):
            row: list[float] = []
            ok = True
            for sid in self._sensor_ids:
                reading = frame.readings.get(sid)
                if reading is None or np.isnan(reading.value):
                    ok = False
                    break
                row.append(reading.value)
            if ok and len(row) == n_sensors:
                rows.append(row)
                frame_indices.append(frame_idx)

        if not rows:
            return None, []
        return np.array(rows, dtype=np.float64), frame_indices

    @staticmethod
    def _fit_gaussian(
        samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
        """Fit (μ, Σ) to samples; return (μ, Σ, Σ⁻¹ or None, log|Σ|)."""
        mu = samples.mean(axis=0)
        cov = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))

        # Trace-scaled ridge keeps the regularization meaningful across unit ranges
        ridge = _COV_RIDGE * max(1.0, float(np.trace(cov)) / cov.shape[0])
        cov = cov + ridge * np.eye(cov.shape[0])

        try:
            cov_inv = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return mu, cov, None, 0.0
        except np.linalg.LinAlgError:
            return mu, cov, None, 0.0

        return mu, cov, cov_inv, float(log_det)

    @staticmethod
    def _kl_to_baseline(
        samples: np.ndarray,
        mu_b: np.ndarray,
        cov_b_inv: np.ndarray,
        log_det_b: float,
    ) -> float:
        """KL(detect ‖ baseline) for two multivariate Gaussians.

        KL(p‖q) = ½ · [log|Σ_q|/|Σ_p| + tr(Σ_q⁻¹ Σ_p)
                       + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − k]
        with p=detect, q=baseline.
        """
        mu_p, cov_p, cov_p_inv, log_det_p = MdiDetector._fit_gaussian(samples)
        if cov_p_inv is None:
            return 0.0

        k = mu_p.shape[0]
        diff = mu_b - mu_p

        try:
            tr_term = float(np.trace(cov_b_inv @ cov_p))
            mahal_term = float(diff.T @ cov_b_inv @ diff)
        except np.linalg.LinAlgError:
            return 0.0

        kl = 0.5 * ((log_det_b - log_det_p) + tr_term + mahal_term - k)
        # KL is non-negative analytically; numerical noise can produce tiny negatives
        return max(kl, 0.0)

    def _scores_to_events(
        self,
        window: TelemetryWindow,
        frame_indices: list[int],
        scores: list[tuple[int, float]],
    ) -> list[AnomalyEvent]:
        """Group consecutive above-threshold scores into intervals; emit one
        AnomalyEvent per run, anchored at the run's peak."""
        events: list[AnomalyEvent] = []
        run: list[tuple[int, float]] = []

        for matrix_idx, kl in scores:
            if kl > self._threshold:
                run.append((matrix_idx, kl))
            else:
                if run:
                    events.append(self._make_event(window, frame_indices, run))
                    run = []
        if run:
            events.append(self._make_event(window, frame_indices, run))

        return events

    def _make_event(
        self,
        window: TelemetryWindow,
        frame_indices: list[int],
        run: list[tuple[int, float]],
    ) -> AnomalyEvent:
        peak_matrix_idx, peak_score = max(run, key=lambda x: x[1])
        first_matrix_idx = run[0][0]
        last_matrix_idx = run[-1][0]

        last_window_end_idx = min(
            last_matrix_idx + self._detect - 1,
            len(frame_indices) - 1,
        )

        peak_frame_idx = frame_indices[peak_matrix_idx]
        first_frame_idx = frame_indices[first_matrix_idx]
        last_frame_idx = frame_indices[last_window_end_idx]

        return AnomalyEvent(
            detector_name=self.name,
            timestamp=window.frames[peak_frame_idx].timestamp,
            affected_sensors=list(self._sensor_ids),
            score=float(peak_score),
            details={
                "interval_start": window.frames[first_frame_idx].timestamp.isoformat(),
                "interval_end": window.frames[last_frame_idx].timestamp.isoformat(),
                "baseline_window_size": self._baseline,
                "detection_window_size": self._detect,
                "n_above_threshold": len(run),
                "kl_peak": float(peak_score),
            },
        )
