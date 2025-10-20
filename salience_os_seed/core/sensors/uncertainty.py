"""Uncertainty sensor measuring short-horizon entropy signals.

The sensor consumes either raw token logits or precomputed entropy estimates
from the runtime state. It maintains a small trailing buffer to smooth spikes
and emits both the instantaneous entropy and a delta-versus-history metadata
payload. Downstream modules primarily use the normalised entropy; the raw value
remains accessible for diagnostics.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Mapping, MutableMapping

import numpy as np

from .base import MedianMADNormalizer, Sensor


class UncertaintySensor(Sensor):
    """Estimate uncertainty via next-token entropy."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        smoothing_window: int = 16,
    ) -> None:
        super().__init__(name="uncertainty", domain="uncertainty", normaliser=normaliser)
        if smoothing_window < 4:
            raise ValueError("smoothing_window must be >= 4 for stability")
        self._window = smoothing_window
        self._buffer: Deque[float] = deque(maxlen=smoothing_window)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        logits = _extract_logits(state)
        if logits is None:
            entropy = float(state.get("prediction", {}).get("entropy_estimate", 0.0))
        else:
            entropy = _entropy_from_logits(logits)
        self._buffer.append(entropy)
        if len(self._buffer) < 4:
            # wait for buffer to fill a bit before smoothing
            return entropy
        return float(np.mean(list(self._buffer)))

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        baseline = float(np.median(self._buffer)) if self._buffer else raw_value
        delta = raw_value - baseline
        return {
            "raw_entropy": raw_value,
            "smooth_delta": delta,
            "buffer_size": float(len(self._buffer)),
            "normalised": norm_value,
        }


def _extract_logits(state: Mapping[str, object]) -> np.ndarray | None:
    prediction = state.get("prediction")
    if not isinstance(prediction, Mapping):
        return None
    logits = prediction.get("token_logits")
    if logits is None:
        return None
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 2:
        # take most recent step if a sequence is provided
        arr = arr[-1]
    return arr


def _entropy_from_logits(logits: np.ndarray) -> float:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / math.log(2.0)  # bits
