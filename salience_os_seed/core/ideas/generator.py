"""Idea generator producing novelty-weighted subgoal proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np

from ..memory import StructuredMemory


@dataclass(frozen=True)
class IdeaProposal:
    """Represents a candidate subgoal or alternative plan."""

    text: str
    novelty: float
    roi: float
    salience_snapshot: Mapping[str, float]


@dataclass
class IdeaFactoryConfig:
    """Configuration for the idea generator."""

    max_ideas: int = 3
    novelty_threshold: float = 0.6
    alignment_threshold: float = 0.5
    drag_ceiling: float = 0.3
    rng_seed: int = 1234


class IdeaGenerator:
    """Generate candidate ideas when NEW and AIM are high."""

    def __init__(self, config: IdeaFactoryConfig) -> None:
        self.config = config
        self.random = np.random.default_rng(config.rng_seed)

    def should_generate(self, salience: Mapping[str, float]) -> bool:
        novelty = float(salience.get("novelty", 0.0))
        alignment = float(salience.get("alignment", 0.0))
        drag = float(salience.get("drag", 0.0))
        return (
            novelty >= self.config.novelty_threshold
            and alignment >= self.config.alignment_threshold
            and drag <= self.config.drag_ceiling
        )

    def generate(
        self,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        memory: StructuredMemory,
        inspirations: Sequence[str] | None = None,
    ) -> List[IdeaProposal]:
        if not self.should_generate(salience):
            return []
        base_candidates = list(self._collect_memory_fragments(memory))
        if inspirations:
            base_candidates.extend(inspirations)
        if not base_candidates:
            base_candidates.append("Draft explicit plan for next 3 steps")
        scores = self._score_candidates(base_candidates, salience, meta_snapshot)
        scored = sorted(scores, key=lambda item: item[1], reverse=True)
        ideas: List[IdeaProposal] = []
        for text, roi in scored[: self.config.max_ideas]:
            novelty = float(salience.get("novelty", 0.0)) + self.random.normal(0, 0.05)
            ideas.append(
                IdeaProposal(
                    text=text,
                    novelty=float(np.clip(novelty, 0.0, 2.0)),
                    roi=float(roi),
                    salience_snapshot=dict(salience),
                )
            )
        return ideas

    def _collect_memory_fragments(self, memory: StructuredMemory) -> Iterable[str]:
        for table in (memory.hypotheses, memory.todos):
            for record in table.iter():
                yield record.text

    def _score_candidates(
        self,
        candidates: Sequence[str],
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
    ) -> List[tuple[str, float]]:
        novelty = float(salience.get("novelty", 0.0))
        alignment = float(salience.get("alignment", 0.0))
        roi = float(meta_snapshot.get("roi", 0.0))
        confidence_gap = 1.0 - float(meta_snapshot.get("confidence", 0.0))
        scores: List[tuple[str, float]] = []
        for candidate in candidates:
            hash_bias = abs(hash(candidate)) % 1000 / 1000.0
            reward = 0.4 * novelty + 0.3 * alignment + 0.2 * confidence_gap + 0.1 * roi + 0.2 * hash_bias
            scores.append((candidate, reward))
        return scores
