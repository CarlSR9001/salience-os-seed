"""Reasoning pattern library for reusable chain-of-thought strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math


@dataclass
class ReasoningPattern:
    name: str
    description: str
    trigger_conditions: Mapping[str, Tuple[float, float]]
    steps: Sequence[str]
    success_rate: float = 0.0
    usage_count: int = 0
    roi: float = 0.0

    def matches(self, salience: Mapping[str, float]) -> bool:
        for key, (lower, upper) in self.trigger_conditions.items():
            value = float(salience.get(key, 0.0))
            if value < lower or value > upper:
                return False
        return True

    def register_outcome(self, success: bool, benefit: float, cost: float) -> None:
        self.usage_count += 1
        if success:
            self.success_rate = ((self.success_rate * (self.usage_count - 1)) + 1.0) / self.usage_count
        else:
            self.success_rate = ((self.success_rate * (self.usage_count - 1)) + 0.0) / self.usage_count
        if cost <= 0:
            cost = 1e-3
        delta_roi = (benefit - cost) / cost
        self.roi = ((self.roi * (self.usage_count - 1)) + delta_roi) / self.usage_count


class PatternLibrary:
    """Curated collection of reasoning strategies."""

    def __init__(self) -> None:
        self.patterns: List[ReasoningPattern] = []
        self.bootstrap_baseline_patterns()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, salience: Mapping[str, float], top_k: int = 5) -> List[ReasoningPattern]:
        scored: List[Tuple[float, ReasoningPattern]] = []
        for pattern in self.patterns:
            if pattern.matches(salience):
                score = pattern.success_rate + 0.5 * pattern.roi
                scored.append((score, pattern))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [pattern for _, pattern in scored[:top_k]]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def extract_from_trace(
        self,
        trace: Sequence[str],
        outcome: bool,
        salience_context: Optional[Mapping[str, float]] = None,
        benefit: float = 1.0,
        cost: float = 1.0,
    ) -> Optional[ReasoningPattern]:
        if not trace:
            return None
        name = f"trace_{len(self.patterns)+1:04d}"
        description = trace[0] if trace else "Extracted reasoning pattern"
        trigger = {
            key: (max(0.0, val - 0.1), min(1.0, val + 0.1))
            for key, val in (salience_context or {}).items()
        }
        steps = tuple(trace[:8])
        pattern = ReasoningPattern(
            name=name,
            description=description,
            trigger_conditions=trigger,
            steps=steps,
        )
        pattern.register_outcome(outcome, benefit=benefit, cost=cost)
        self.patterns.append(pattern)
        return pattern

    def prune(self, min_roi: float = 0.1, min_usage: int = 3) -> None:
        self.patterns = [p for p in self.patterns if (p.usage_count >= min_usage and p.roi >= min_roi)]

    def log_usage(
        self,
        pattern: ReasoningPattern,
        success: bool,
        benefit: float,
        cost: float,
    ) -> None:
        pattern.register_outcome(success, benefit=benefit, cost=cost)

    # ------------------------------------------------------------------
    # Baseline seed patterns
    # ------------------------------------------------------------------
    def bootstrap_baseline_patterns(self) -> None:
        seeds: Iterable[ReasoningPattern] = [
            ReasoningPattern(
                name="decompose_complex",
                description="Break complex goals into manageable subgoals",
                trigger_conditions={"aim": (0.6, 1.0), "key": (0.6, 1.0)},
                steps=(
                    "Enumerate subgoals",
                    "Solve each subgoal",
                    "Aggregate partial solutions",
                ),
                success_rate=0.7,
                usage_count=10,
                roi=0.4,
            ),
            ReasoningPattern(
                name="verify_then_continue",
                description="Pause to verify intermediate results",
                trigger_conditions={"uncertainty": (0.5, 1.0)},
                steps=(
                    "Identify assumption",
                    "Check assumption against memory",
                    "Adjust plan based on verification",
                ),
                success_rate=0.6,
                usage_count=8,
                roi=0.35,
            ),
            ReasoningPattern(
                name="retrieve_similar",
                description="Query memory for analogous situations",
                trigger_conditions={"novelty": (0.6, 1.0), "drag": (0.0, 0.4)},
                steps=(
                    "Search structured memory",
                    "Select most similar case",
                    "Adapt prior solution",
                ),
                success_rate=0.65,
                usage_count=9,
                roi=0.3,
            ),
            ReasoningPattern(
                name="meta_check",
                description="Re-evaluate assumptions when verification fails",
                trigger_conditions={"confidence": (0.5, 1.0), "verification": (0.0, 0.3)},
                steps=(
                    "List assumptions",
                    "Identify weak links",
                    "Revise plan",
                ),
                success_rate=0.55,
                usage_count=6,
                roi=0.25,
            ),
            ReasoningPattern(
                name="simplify_approach",
                description="Reduce problem scope when drag is high",
                trigger_conditions={"drag": (0.5, 1.0), "progress": (0.0, 0.4)},
                steps=(
                    "Identify blocking factor",
                    "Remove or postpone blocker",
                    "Resume simplified task",
                ),
                success_rate=0.58,
                usage_count=7,
                roi=0.28,
            ),
        ]
        self.patterns.extend(seeds)
