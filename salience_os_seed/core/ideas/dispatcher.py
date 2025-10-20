"""Dispatcher for persisting accepted ideas into structured memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..memory import StructuredMemory
from .generator import IdeaProposal
from .simulator import SimulationResult


def default_roi_threshold(meta_snapshot: Mapping[str, float]) -> float:
    base = 0.2
    roi_bias = float(meta_snapshot.get("roi", 0.0))
    return base + 0.1 * roi_bias


@dataclass
class IdeaDispatcher:
    """Persist idea proposals with ROI above threshold into todos[]."""

    memory: StructuredMemory

    def dispatch(
        self,
        simulations: Sequence[SimulationResult],
        meta_snapshot: Mapping[str, float],
        roi_threshold: float | None = None,
    ) -> list[IdeaProposal]:
        threshold = roi_threshold if roi_threshold is not None else default_roi_threshold(meta_snapshot)
        accepted: list[IdeaProposal] = []
        for result in simulations:
            if result.roi >= threshold:
                self.memory.todos.add(result.proposal.text, score=result.roi)
                accepted.append(result.proposal)
        return accepted
