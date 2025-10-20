"""Bandit trainer updating controller weights from verification returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

from .actions import ControllerAction


@dataclass
class BanditConfig:
    """Configuration for the bandit update routine."""

    learning_rate: float = 0.05
    reward_scale: float = 1.0
    penalty_scale: float = 1.0
    max_abs_reward: float = 5.0


class BanditTrainer:
    """Applies bandit-style updates to the controller weight store."""

    def __init__(self, config: BanditConfig, weight_store: MutableMapping[str, MutableMapping[str, float]]) -> None:
        self.config = config
        self.weight_store = weight_store

    def update(self, action: ControllerAction, reward: float) -> None:
        cfg = self.config
        clamped = max(-cfg.max_abs_reward, min(cfg.max_abs_reward, reward))
        scaled_reward = (cfg.reward_scale if clamped >= 0 else cfg.penalty_scale) * clamped
        bucket = self.weight_store.setdefault(self._key(action), {"bias": 0.0, "count": 0.0})
        bias = bucket.get("bias", 0.0)
        updated_bias = bias + cfg.learning_rate * (scaled_reward - bias)
        bucket["bias"] = updated_bias
        bucket["count"] = bucket.get("count", 0.0) + 1.0

    def snapshot(self) -> Mapping[str, Mapping[str, float]]:
        return {key: dict(value) for key, value in self.weight_store.items()}

    @staticmethod
    def _key(action: ControllerAction) -> str:
        return f"depth={action.cot_depth}|op={action.operator.name}|patch={action.patch.name}"
