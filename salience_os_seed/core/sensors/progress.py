"""Progress/KEY sensor forecasting remaining work.

The sensor estimates KEY: the expected unlocking power or predicted delta in
steps remaining if we successfully execute the current action plan. It does so
by combining a lightweight progress head (predicting remaining steps) with
observed deltas over time, effectively forming a one-step temporal difference.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, MutableMapping

import numpy as np

from .base import MedianMADNormalizer, Sensor


class ProgressSensor(Sensor):
    """Predict Δ(steps remaining) → KEY."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        smoothing_window: int = 12,
    ) -> None:
        super().__init__(name="progress", domain="progress", normaliser=normaliser)
        if smoothing_window < 6:
            raise ValueError("smoothing_window must be >= 6 for progress sensor")
        self._window = smoothing_window
        self._pred_buffer: Deque[float] = deque(maxlen=smoothing_window)
        self._delta_buffer: Deque[float] = deque(maxlen=smoothing_window)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        prediction = state.get("prediction")
        if not isinstance(prediction, Mapping):
            return 0.0
        steps_remaining = float(prediction.get("steps_remaining", 0.0))
        prev = float(memory.get("last_steps_remaining", steps_remaining))
        delta = prev - steps_remaining
        self._pred_buffer.append(steps_remaining)
        self._delta_buffer.append(delta)
        memory_mut = getattr(memory, "setdefault", None)
        if callable(memory_mut):
            memory.setdefault("last_steps_remaining", steps_remaining)
        if len(self._delta_buffer) < 3:
            return delta
        expected_delta = float(np.mean(self._delta_buffer))
        return expected_delta

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        avg_steps = float(np.mean(self._pred_buffer)) if self._pred_buffer else 0.0
        return {
            "raw_progress": raw_value,
            "avg_steps_remaining": avg_steps,
            "normalised": norm_value,
        }
