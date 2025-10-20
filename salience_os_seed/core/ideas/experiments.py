"""Self-directed experiment utilities for the idea factory."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional


@dataclass
class SelfExperiment:
    """Description of a runtime experiment."""

    name: str
    hypothesis: str
    parameter_overrides: Mapping[str, float]
    duration_steps: int
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    results: Dict[str, float] = field(default_factory=dict)
    conclusion: Optional[str] = None
    baseline_verification: float = 0.0
    parameter_metadata: Dict[str, str] = field(default_factory=dict)


PARAMETER_ALIASES: Mapping[str, str] = {
    "scheduler.threshold": "scheduler.min_budget_ratio",
}


PARAMETER_BUILDERS: Mapping[str, Callable[[float, float], float]] = {
    "controller.lambda_cost": lambda uncertainty, _drag: 0.8 + 0.4 * uncertainty,
    "scheduler.min_budget_ratio": lambda _uncertainty, drag: 0.5 + 0.3 * (1.0 - drag),
}


PARAMETER_LABELS: Mapping[str, str] = {
    "controller.lambda_cost": "lambda_cost",
    "scheduler.min_budget_ratio": "scheduler",
}


class ExperimentDispatcher:
    """Proposes and manages self-directed runtime experiments."""

    def __init__(self, allowed_parameters: Iterable[str], max_concurrent: int = 1) -> None:
        allowed = list(dict.fromkeys(allowed_parameters))
        if not allowed:
            raise ValueError("allowed_parameters must contain at least one entry")
        self.allowed_parameters = allowed
        self.max_concurrent = max(1, max_concurrent)
        self.active: List[SelfExperiment] = []

    def propose_experiment(
        self,
        salience: Mapping[str, float],
        verification_success: Optional[bool],
    ) -> Optional[SelfExperiment]:
        if len(self.active) >= self.max_concurrent:
            return None
        verification = 1.0 if verification_success else 0.0 if verification_success is False else 0.5
        uncertainty = float(salience.get("uncertainty", 0.0))
        drag = float(salience.get("drag", 0.0))
        if verification >= 0.6 or uncertainty < 0.4 or drag > 0.7:
            return None
        experiment = self._build_experiment(uncertainty, drag)
        self.active.append(experiment)
        return experiment

    def _canonical_parameter(self, parameter: str) -> str:
        return PARAMETER_ALIASES.get(parameter, parameter)

    def _build_experiment(self, uncertainty: float, drag: float) -> SelfExperiment:
        overrides: Dict[str, float] = {}
        metadata: Dict[str, str] = {}
        name_parts: List[str] = []
        for parameter in self.allowed_parameters:
            canonical = self._canonical_parameter(parameter)
            metadata[parameter] = canonical
            builder = PARAMETER_BUILDERS.get(canonical)
            if not builder or canonical in overrides:
                continue
            overrides[canonical] = builder(uncertainty, drag)
            name_parts.append(PARAMETER_LABELS.get(canonical, canonical))
        if not overrides:
            fallback_alias = self.allowed_parameters[0]
            fallback = self._canonical_parameter(fallback_alias)
            metadata.setdefault(fallback_alias, fallback)
            overrides[fallback] = 0.5
            name_parts.append(PARAMETER_LABELS.get(fallback, fallback))
        name = "exp::" + ":".join(name_parts)
        return SelfExperiment(
            name=name,
            hypothesis="Adjust parameters to improve verification success",
            parameter_overrides=overrides,
            duration_steps=16,
            parameter_metadata=metadata,
        )

    def run_experiment(
        self,
        experiment: SelfExperiment,
        step_fn: Callable[[Mapping[str, float]], Mapping[str, float]],
    ) -> SelfExperiment:
        for _ in range(experiment.duration_steps):
            metrics = step_fn(experiment.parameter_overrides)
            for key, value in metrics.items():
                experiment.metrics.setdefault(key, []).append(float(value))
        return experiment

    def analyze_results(self, experiment: SelfExperiment) -> SelfExperiment:
        stats: Dict[str, float] = {}
        for key, values in experiment.metrics.items():
            if not values:
                continue
            stats[f"{key}_mean"] = statistics.fmean(values)
            if len(values) > 1:
                stats[f"{key}_stdev"] = statistics.pstdev(values)
        experiment.results = stats
        experiment.conclusion = self._conclude(stats, experiment.baseline_verification)
        if experiment in self.active:
            self.active.remove(experiment)
        return experiment

    def _conclude(self, stats: Mapping[str, float], baseline: float) -> str:
        if not stats:
            return "insufficient data"
        improvement = stats.get("verification_mean", 0.0) - baseline
        if improvement > 0.05:
            return "positive"
        if improvement < -0.05:
            return "negative"
        return "neutral"
