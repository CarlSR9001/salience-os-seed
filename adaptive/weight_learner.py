"""Simplified adaptive weight learner for the seed runtime.

This learner adjusts salience weighting coefficients based on aggregated metrics
and gradient feedback. It does not depend on the Enhanced SUA agent stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional


@dataclass
class SalienceWeights:
    novelty: float
    retention: float
    payoff: float

    def normalized(self) -> "SalienceWeights":
        total = self.novelty + self.retention + self.payoff
        if total <= 0:
            return SalienceWeights(1 / 3, 1 / 3, 1 / 3)
        return SalienceWeights(
            novelty=self.novelty / total,
            retention=self.retention / total,
            payoff=self.payoff / total,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "novelty": float(self.novelty),
            "retention": float(self.retention),
            "payoff": float(self.payoff),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, float]) -> "SalienceWeights":
        return cls(
            novelty=float(payload.get("novelty", 1 / 3)),
            retention=float(payload.get("retention", 1 / 3)),
            payoff=float(payload.get("payoff", 1 / 3)),
        )


class AdaptiveWeightLearner:
    def __init__(
        self,
        weights: Dict[str, SalienceWeights],
        *,
        stability_limit: float = 0.08,
        cooldown_seconds: float = 90.0,
        persist_callback: Optional[Callable[[Dict[str, SalienceWeights]], None]] = None,
    ) -> None:
        self._weights = {task: value.normalized() for task, value in weights.items()}
        self._history: Dict[str, list[dict[str, float]]] = {task: [] for task in weights}
        self._last_update = 0.0
        self._stability_limit = stability_limit
        self._cooldown = cooldown_seconds
        self._persist = persist_callback

    def consider_update(
        self,
        task: str,
        metrics: Dict[str, float],
        gradients: Optional[Dict[str, float]] = None,
        *,
        suggestions: Optional[Iterable[str]] = None,
    ) -> Optional[SalienceWeights]:
        now = time.time()
        if now - self._last_update < self._cooldown:
            return None
        current = self._weights.setdefault(task, SalienceWeights(1 / 3, 1 / 3, 1 / 3))
        novelty_adj = gradients.get("novelty", 0.0) if gradients else 0.0
        retention_adj = gradients.get("retention", 0.0) if gradients else 0.0
        payoff_adj = gradients.get("payoff", 0.0) if gradients else 0.0

        if suggestions:
            for item in suggestions:
                item_lower = item.lower()
                if "novel" in item_lower:
                    novelty_adj += 0.03
                if "remember" in item_lower or "retain" in item_lower:
                    retention_adj += 0.03
                if "payoff" in item_lower or "immediate" in item_lower:
                    payoff_adj += 0.03

        quality = metrics.get("quality", 0.5)
        if quality < 0.4:
            payoff_adj += 0.02
        elif quality > 0.8:
            retention_adj += 0.02

        proposal = SalienceWeights(
            novelty=self._clamp(current.novelty + novelty_adj),
            retention=self._clamp(current.retention + retention_adj),
            payoff=self._clamp(current.payoff + payoff_adj),
        ).normalized()
        if not self._stable(current, proposal):
            return None

        self._weights[task] = proposal
        self._history.setdefault(task, []).append({"time": now, "quality": quality})
        self._last_update = now
        if self._persist:
            try:
                self._persist(self._weights)
            except Exception:
                pass
        return proposal

    def weights(self) -> Dict[str, SalienceWeights]:
        return {task: value for task, value in self._weights.items()}

    def history(self, task: Optional[str] = None) -> Dict[str, list[dict[str, float]]]:
        if task is None:
            return {key: list(value) for key, value in self._history.items()}
        return {task: list(self._history.get(task, ())) }

    def _stable(self, old: SalienceWeights, new: SalienceWeights) -> bool:
        return (
            abs(new.novelty - old.novelty) <= self._stability_limit
            and abs(new.retention - old.retention) <= self._stability_limit
            and abs(new.payoff - old.payoff) <= self._stability_limit
        )

    @staticmethod
    def _clamp(value: float, lower: float = 0.05, upper: float = 0.9) -> float:
        return max(lower, min(upper, value))

    def state_dict(self) -> Dict[str, object]:
        return {
            "weights": {task: weights.to_dict() for task, weights in self._weights.items()},
            "last_update": self._last_update,
            "history": {task: list(entries) for task, entries in self._history.items()},
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        weights_payload = payload.get("weights", {})
        restored: Dict[str, SalienceWeights] = {}
        if isinstance(weights_payload, dict):
            for task, data in weights_payload.items():
                restored[task] = SalienceWeights.from_dict(dict(data)).normalized()
        if restored:
            self._weights = restored
        last_update = payload.get("last_update")
        if isinstance(last_update, (int, float)):
            self._last_update = float(last_update)
        history_payload = payload.get("history", {})
        if isinstance(history_payload, dict):
            self._history = {
                task: list(entries)
                for task, entries in history_payload.items()
                if isinstance(entries, list)
            }
