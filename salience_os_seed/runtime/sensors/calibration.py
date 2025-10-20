"""Probe models that calibrate raw sensor outputs against runtime outcomes.

The calibration suite maintains lightweight linear probes per salience channel.
Each probe collects (raw sensor value, target weight, reward, success) tuples
from the running system.  When enough data is accumulated the probe solves a
ridge-regularised least squares system to map raw sensor readings to weight
suggestions in ``[0, 1]``.  The resulting weights are blended with heuristic
priors upstream (see :mod:`salience_os_seed.adaptive.coordinator`).

The implementation intentionally favours simplicity: probes operate on small
rolling windows and avoid any dependency on heavy ML frameworks.  They provide a
compact summary that can be persisted inside runtime checkpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, MutableMapping, Tuple

import numpy as np

__all__ = [
    "CalibrationOutcome",
    "ProbeModel",
    "SensorCalibrationSuite",
]


@dataclass(slots=True)
class CalibrationOutcome:
    """Single supervision example for a sensor calibration probe."""

    raw_value: float
    target_weight: float
    reward: float
    success: float


@dataclass(slots=True)
class ProbeModel:
    """Linear probe with ridge regularisation for a single sensor channel."""

    name: str
    baseline_weight: float
    window: int = 128
    ridge_penalty: float = 1e-2
    min_samples: int = 8
    _history: Deque[CalibrationOutcome] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.window)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def register(self, outcome: CalibrationOutcome) -> None:
        """Add an outcome to the rolling history."""

        self._history.append(outcome)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, raw_value: float) -> Tuple[float, float]:
        """Return a calibrated weight and confidence for the provided raw value."""

        baseline = self._baseline_map(raw_value)
        if len(self._history) < self.min_samples:
            return baseline, 0.0

        x = np.asarray([entry.raw_value for entry in self._history], dtype=np.float64)
        y = np.asarray([entry.target_weight for entry in self._history], dtype=np.float64)
        # Successes/rewards bias the regression toward reliable measurements.
        w = np.asarray(
            [1.0 + 0.5 * max(0.0, entry.reward) + 0.5 * max(0.0, entry.success) for entry in self._history],
            dtype=np.float64,
        )
        design = np.vstack([x, np.ones_like(x)]).T
        ridge_matrix = self.ridge_penalty * np.eye(2)
        try:
            lhs = design.T @ (w[:, None] * design) + ridge_matrix
            rhs = design.T @ (w * y)
            coeffs = np.linalg.solve(lhs, rhs)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            estimate = slope * float(raw_value) + intercept
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            estimate = baseline
        calibrated = float(np.clip(estimate, 0.05, 0.95))
        # Confidence increases with sample count and falls back to baseline otherwise.
        confidence = float(min(1.0, len(self._history) / (self.min_samples * 1.5)))
        return calibrated, confidence

    def _baseline_map(self, raw_value: float) -> float:
        centred = float(np.tanh(raw_value))
        heuristic = self.baseline_weight + 0.25 * centred
        return float(np.clip(heuristic, 0.05, 0.95))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Mapping[str, float]:
        """Expose recent statistics for debugging/telemetry."""

        if not self._history:
            return {
                "samples": 0,
                "mean_raw": 0.0,
                "mean_weight": self.baseline_weight,
            }
        raw = np.asarray([entry.raw_value for entry in self._history], dtype=np.float64)
        weights = np.asarray([entry.target_weight for entry in self._history], dtype=np.float64)
        return {
            "samples": float(len(self._history)),
            "mean_raw": float(raw.mean()),
            "mean_weight": float(weights.mean()),
        }


class SensorCalibrationSuite:
    """Manage calibration probes for multiple salience channels."""

    def __init__(
        self,
        baselines: Mapping[str, float],
        *,
        history_window: int = 128,
        ridge_penalty: float = 1e-2,
        min_samples: int = 8,
    ) -> None:
        self._probes: MutableMapping[str, ProbeModel] = {
            name: ProbeModel(
                name=name,
                baseline_weight=float(weight),
                window=history_window,
                ridge_penalty=ridge_penalty,
                min_samples=min_samples,
            )
            for name, weight in baselines.items()
        }

    def observe(
        self,
        channel: str,
        raw_value: float,
        target_weight: float,
        *,
        reward: float,
        success: float,
    ) -> None:
        probe = self._probes.get(channel)
        if probe is None:
            return
        outcome = CalibrationOutcome(
            raw_value=float(raw_value),
            target_weight=float(np.clip(target_weight, 0.0, 1.0)),
            reward=float(reward),
            success=float(np.clip(success, 0.0, 1.0)),
        )
        probe.register(outcome)

    def predict(self, raw_inputs: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return calibrated weights + confidences for the provided raw inputs."""

        weights: Dict[str, float] = {}
        confidences: Dict[str, float] = {}
        for name, value in raw_inputs.items():
            probe = self._probes.get(name)
            if probe is None:
                continue
            weight, confidence = probe.predict(float(value))
            weights[name] = weight
            confidences[name] = confidence
        return weights, confidences

    def snapshot(self) -> Mapping[str, Mapping[str, float]]:
        return {name: probe.snapshot() for name, probe in self._probes.items()}
