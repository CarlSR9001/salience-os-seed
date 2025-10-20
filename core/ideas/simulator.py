"""Idea simulator estimating ROI for proposed ideas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np

from .generator import IdeaProposal


@dataclass
class SimulationResult:
    """Outcome of a lightweight idea simulation."""

    proposal: IdeaProposal
    predicted_gain: float
    predicted_cost: float
    roi: float


class IdeaSimulator:
    """Perform quick heuristics to estimate idea payoff."""

    def __init__(self, cost_per_step: float = 20.0) -> None:
        self.cost_per_step = cost_per_step

    def simulate(
        self,
        proposals: Sequence[IdeaProposal],
        meta_snapshot: Mapping[str, float],
        salience: Mapping[str, float],
    ) -> List[SimulationResult]:
        difficulty = float(meta_snapshot.get("difficulty", 1.0))
        confidence = float(meta_snapshot.get("confidence", 0.0))
        drag = float(salience.get("drag", 0.0))
        uncertainty = float(salience.get("uncertainty", 0.0))
        results: List[SimulationResult] = []
        for proposal in proposals:
            gain = self._estimate_gain(proposal, difficulty, uncertainty)
            cost = self._estimate_cost(proposal, drag, confidence)
            roi = gain - cost / max(self.cost_per_step, 1.0)
            results.append(
                SimulationResult(
                    proposal=proposal,
                    predicted_gain=gain,
                    predicted_cost=cost,
                    roi=roi,
                )
            )
        return results

    def _estimate_gain(self, proposal: IdeaProposal, difficulty: float, uncertainty: float) -> float:
        novelty_boost = 0.6 * proposal.novelty
        difficulty_term = 0.3 * np.tanh(difficulty)
        uncertainty_term = 0.1 * np.tanh(uncertainty)
        return float(novelty_boost + difficulty_term + uncertainty_term)

    def _estimate_cost(self, proposal: IdeaProposal, drag: float, confidence: float) -> float:
        drag_penalty = 1.0 + max(drag, 0.0)
        confidence_discount = 1.0 - 0.3 * confidence
        base_cost = self.cost_per_step * confidence_discount
        return float(base_cost * drag_penalty)
