"""Adaptive subsystems for the Salience OS seed runtime."""

from .coordinator import AdaptiveCoordinator, GatingSummary
from .vault import AdaptiveVault, VaultStats, WeightProvenance, WeightSnapshot
from .gradient_flow import AdaptiveGradientFlow, FlowEstimate, FlowSignal
from .weight_learner import AdaptiveWeightLearner, SalienceWeights
from .truth import TruthGuard, TruthConfig, TruthDecision
from .elegance import EleganceJudge, EleganceCandidate, EleganceConfig, EleganceMetrics, EleganceResult
from .axioms import AxiomGuard, AxiomSet, Axiom, AxiomViolation

__all__ = [
    "AdaptiveCoordinator",
    "GatingSummary",
    "AdaptiveVault",
    "VaultStats",
    "WeightProvenance",
    "WeightSnapshot",
    "AdaptiveGradientFlow",
    "FlowEstimate",
    "FlowSignal",
    "AdaptiveWeightLearner",
    "SalienceWeights",
    "TruthGuard",
    "TruthConfig",
    "TruthDecision",
    "EleganceJudge",
    "EleganceCandidate",
    "EleganceConfig",
    "EleganceMetrics",
    "EleganceResult",
    "AxiomGuard",
    "AxiomSet",
    "Axiom",
    "AxiomViolation",
]
