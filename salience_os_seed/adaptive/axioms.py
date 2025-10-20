"""Axiomatic guardrails adapted for the seed runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class Axiom:
    id: str
    rule: str
    description: str
    tags: Iterable[str] = field(default_factory=tuple)


@dataclass
class AxiomViolation:
    axiom_id: str
    reason: str
    context: Dict[str, object]


@dataclass
class AxiomSet:
    axioms: Dict[str, Axiom]

    @classmethod
    def default(cls) -> "AxiomSet":
        axioms = {
            "SALIENCE_MONOTONE_AIM": Axiom(
                id="SALIENCE_MONOTONE_AIM",
                rule="dS/dAIM >= 0",
                description="Aim-driven salience adjustments must not reduce overall salience.",
                tags=("salience", "monotonic"),
            ),
            "TRUTH_GATE_SPEAK": Axiom(
                id="TRUTH_GATE_SPEAK",
                rule="speak iff T_star >= 0.85 and combined_score >= 1.2",
                description="Only speak when truth star and combined salience exceed thresholds.",
                tags=("truth", "speak"),
            ),
            "SAFETY_COPY_ON_WRITE": Axiom(
                id="SAFETY_COPY_ON_WRITE",
                rule="writes must be copy-on-write",
                description="Prevent destructive modification of base knowledge stores.",
                tags=("safety", "memory"),
            ),
        }
        return cls(axioms=axioms)


class AxiomGuard:
    def __init__(self, axiom_set: AxiomSet | None = None) -> None:
        self._axioms = axiom_set or AxiomSet.default()

    def evaluate_speak(self, *, truth_star: float, combined_score: float) -> List[AxiomViolation]:
        violations: List[AxiomViolation] = []
        axiom = self._axioms.axioms.get("TRUTH_GATE_SPEAK")
        if axiom and (truth_star < 0.85 or combined_score < 1.2):
            violations.append(
                AxiomViolation(
                    axiom_id=axiom.id,
                    reason="truth_star_or_score_below_threshold",
                    context={"truth_star": truth_star, "combined_score": combined_score},
                )
            )
        return violations

    def evaluate_salience_shift(self, delta_salience: float) -> List[AxiomViolation]:
        violations: List[AxiomViolation] = []
        axiom = self._axioms.axioms.get("SALIENCE_MONOTONE_AIM")
        if axiom and delta_salience < 0:
            violations.append(
                AxiomViolation(
                    axiom_id=axiom.id,
                    reason="salience_decreased",
                    context={"delta_salience": delta_salience},
                )
            )
        return violations

    def all_axioms(self) -> Dict[str, Axiom]:
        return dict(self._axioms.axioms)
