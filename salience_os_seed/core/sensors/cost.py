"""Operator cost forecasting sensor.

The cost sensor predicts the expected latency/token cost for the candidate
operator/action about to be executed. It reads the most recent action proposal
from runtime state (`state["decision_proposal"]`) and feeds it through a
lightweight regressor head whose coefficients are maintained in memory. The
regressor is a simple dictionary keyed by operator names and depth that stores
empirically observed costs. Exponential moving averages keep the estimates
current without reacting too aggressively to outliers.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping

from .base import MedianMADNormalizer, Sensor


class CostSensor(Sensor):
    """Predict runtime cost for the proposed action."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        ema_decay: float = 0.85,
        default_cost: float = 40.0,
    ) -> None:
        super().__init__(name="cost", domain="cost", normaliser=normaliser)
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        self._ema_decay = ema_decay
        self._default_cost = float(default_cost)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        proposal = state.get("decision_proposal")
        if not isinstance(proposal, Mapping):
            return self._default_cost
        operator = str(proposal.get("operator", "unknown"))
        depth = int(proposal.get("cot_depth", 0))
        features = (operator, depth)
        estimator = _CostEstimator(memory)
        prediction = estimator.predict(features, fallback=self._default_cost)
        return prediction

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        proposal = state.get("decision_proposal") or {}
        operator = proposal.get("operator", "unknown")
        depth = int(proposal.get("cot_depth", 0))
        return {
            "raw_cost": raw_value,
            "operator": float(hash(operator) % 1000),  # hashed for cheap logging
            "cot_depth": float(depth),
            "normalised": norm_value,
        }


class _CostEstimator:
    """Proxy around mutable memory for EMA cost tracking."""

    def __init__(self, memory: Mapping[str, object]) -> None:
        self._registry: MutableMapping[str, MutableMapping[str, float]] = memory.setdefault(  # type: ignore[attr-defined]
            "cost_estimates",
            {},
        )  # type: ignore[assignment]

    def predict(self, features: tuple[str, int], fallback: float) -> float:
        operator, depth = features
        depth_key = f"depth::{depth}"
        bucket = self._registry.setdefault(operator, {})
        return float(bucket.get(depth_key, fallback))

    def update(self, features: tuple[str, int], observed_cost: float, decay: float) -> None:
        operator, depth = features
        depth_key = f"depth::{depth}"
        bucket = self._registry.setdefault(operator, {})
        prev = bucket.get(depth_key, observed_cost)
        bucket[depth_key] = (1.0 - decay) * observed_cost + decay * prev
