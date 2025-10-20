"""Elegance drive components adapted for the seed runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class EleganceCandidate:
    weight_id: str
    description: str
    baseline_bits: float
    operations: Iterable[str]


@dataclass
class EleganceMetrics:
    delta_mdl: float
    regressions: float
    transfer_gain: float
    orthogonal_novelty: float
    tstar_min: float
    risk: float
    validators_passed: int


@dataclass
class EleganceResult:
    candidate: EleganceCandidate
    metrics: EleganceMetrics
    score: float
    accepted: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class EleganceConfig:
    theta_elegance: float = 0.05
    regressions_max: float = 0.15
    tstar_min: float = 0.45
    lambda_regress: float = 0.6
    mu_transfer: float = 0.4
    kappa_novelty: float = 0.3


class EleganceJudge:
    def __init__(self, cfg: EleganceConfig | None = None) -> None:
        self.cfg = cfg or EleganceConfig()
        self.history: List[EleganceResult] = []

    def evaluate(self, candidate: EleganceCandidate, metrics: EleganceMetrics) -> EleganceResult:
        cfg = self.cfg
        score = (
            metrics.delta_mdl
            - cfg.lambda_regress * metrics.regressions
            + cfg.mu_transfer * metrics.transfer_gain
            + cfg.kappa_novelty * metrics.orthogonal_novelty
        )
        accepted = (
            metrics.delta_mdl >= cfg.theta_elegance
            and metrics.regressions <= cfg.regressions_max
            and metrics.tstar_min >= cfg.tstar_min
            and metrics.risk < 1.0
        )
        result = EleganceResult(candidate=candidate, metrics=metrics, score=score, accepted=accepted)
        self.history.append(result)
        self.history = self.history[-128:]
        return result

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {"accepted": 0.0, "rejected": 0.0}
        accepted = sum(1 for item in self.history if item.accepted)
        rejected = len(self.history) - accepted
        return {"accepted": float(accepted), "rejected": float(rejected)}
