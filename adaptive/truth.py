"""Truth-aware gating utilities scoped to the seed runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TruthWeights:
    beta_uncertainty: float = 0.4
    gamma_contradiction: float = 0.35
    delta_risk: float = 0.25


@dataclass
class TruthThresholds:
    tau_block: float = 0.12
    tau_speak: float = 0.45
    tau_save: float = 0.4


@dataclass
class TruthHysteresis:
    speak_on: float = 0.55
    speak_off: float = 0.45
    save_on: float = 0.5
    save_off: float = 0.4


@dataclass
class TruthConfig:
    alpha_truth: float = 0.35
    weights: TruthWeights = field(default_factory=TruthWeights)
    thresholds: TruthThresholds = field(default_factory=TruthThresholds)
    hysteresis: TruthHysteresis = field(default_factory=TruthHysteresis)


@dataclass
class TruthDecision:
    action_score: float
    combined_score: float
    decision: str
    reason: str


class TruthGuard:
    def __init__(self, cfg: TruthConfig | None = None) -> None:
        self.cfg = cfg or TruthConfig()

    def truth_score(self, truth: float, uncertainty: float, contradiction: float, risk: float) -> float:
        weights = self.cfg.weights
        score = truth
        score -= weights.beta_uncertainty * uncertainty
        score -= weights.gamma_contradiction * contradiction
        score -= weights.delta_risk * risk
        return max(0.0, min(1.0, score))

    def fuse(self, salience_score: float, truth_star: float) -> float:
        return salience_score + self.cfg.alpha_truth * truth_star

    def decide(
        self,
        *,
        action_score: float,
        combined_score: float,
        truth_star: float,
        previous: Optional[str] = None,
    ) -> TruthDecision:
        thresholds = self.cfg.thresholds
        hysteresis = self.cfg.hysteresis

        if truth_star < thresholds.tau_block:
            return TruthDecision(action_score, combined_score, "DROP", "truth_star_below_block")

        if truth_star >= thresholds.tau_speak and combined_score >= hysteresis.speak_on:
            return TruthDecision(action_score, combined_score, "SPEAK", "speak_gate_high")
        if previous == "SPEAK" and combined_score >= hysteresis.speak_off:
            return TruthDecision(action_score, combined_score, "SPEAK", "speak_hysteresis")

        if truth_star >= thresholds.tau_save and combined_score >= hysteresis.save_on:
            return TruthDecision(action_score, combined_score, "SAVE", "save_gate_high")
        if previous == "SAVE" and combined_score >= hysteresis.save_off:
            return TruthDecision(action_score, combined_score, "SAVE", "save_hysteresis")

        return TruthDecision(action_score, combined_score, "DROP", "below_truth_or_salience")
