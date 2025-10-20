"""Truth sensor estimating factual confidence for salience."""
from __future__ import annotations

from typing import Mapping, MutableMapping

import numpy as np

from .base import MedianMADNormalizer, Sensor


class TruthSensor(Sensor):
    """Infer a lightweight truth estimate from meta signals."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        confidence_weight: float = 0.6,
        roi_weight: float = 0.3,
        contradiction_weight: float = 0.1,
    ) -> None:
        super().__init__(name="truth", domain="truth", normaliser=normaliser)
        total = confidence_weight + roi_weight + contradiction_weight
        if total <= 0:
            raise ValueError("TruthSensor weights must sum to a positive value")
        self._confidence_weight = confidence_weight / total
        self._roi_weight = roi_weight / total
        self._contradiction_weight = contradiction_weight / total

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        del memory
        confidence = float(meta.get("confidence", 0.0))
        roi = float(meta.get("roi", 0.0))
        verification = float(meta.get("verification_pass_rate", 0.0))
        confidence_scaled = 0.5 * (confidence + 1.0)
        roi_scaled = 0.5 * (roi + 1.0)
        contradiction = float(state.get("contradictions", 0.0))
        contradiction_penalty = 1.0 - min(1.0, contradiction / 3.0)
        verification_boost = np.clip(verification, 0.0, 1.0)
        truth_estimate = (
            self._confidence_weight * np.clip(confidence_scaled, 0.0, 1.0)
            + self._roi_weight * np.clip(roi_scaled, 0.0, 1.0)
            + self._contradiction_weight * contradiction_penalty
        )
        truth_estimate = 0.75 * truth_estimate + 0.25 * verification_boost
        return float(np.clip(truth_estimate, 0.0, 1.0))

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        del state, memory
        return {
            "raw_truth": raw_value,
            "confidence": float(meta.get("confidence", 0.0)),
            "roi": float(meta.get("roi", 0.0)),
            "verification_rate": float(meta.get("verification_pass_rate", 0.0)),
            "normalised": norm_value,
        }
