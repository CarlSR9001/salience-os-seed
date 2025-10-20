"""Self-experiment coordination utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from ..core.ideas import ExperimentDispatcher, SelfExperiment


@dataclass(slots=True)
class ActiveExperiment:
    experiment: SelfExperiment
    remaining_steps: int
    originals: Dict[str, float]


@dataclass(slots=True)
class ExperimentCoordinator:
    dispatcher: ExperimentDispatcher | None
    duration: int

    _active: List[ActiveExperiment] = field(default_factory=list)

    def propose(
        self,
        salience: Mapping[str, float],
        verification_passed: Optional[bool],
        overrides_applier,
    ) -> None:
        if not self.dispatcher:
            return
        proposed = self.dispatcher.propose_experiment(salience, verification_passed)
        if not proposed:
            return
        proposed.duration_steps = max(1, self.duration)
        baseline = 1.0 if verification_passed else 0.0 if verification_passed is False else 0.5
        proposed.baseline_verification = baseline
        originals = overrides_applier(proposed.parameter_overrides)
        self._active.append(
            ActiveExperiment(
                experiment=proposed,
                remaining_steps=proposed.duration_steps,
                originals=originals,
            )
        )

    def advance(
        self,
        salience: Mapping[str, float],
        verification_passed: Optional[bool],
        *,
        overrides_reverter,
    ) -> List[Mapping[str, object]]:
        if not self.dispatcher:
            return []
        reports: List[Mapping[str, object]] = []
        finished: List[ActiveExperiment] = []
        for active in self._active:
            experiment = active.experiment
            verification_value = 1.0 if verification_passed else 0.0 if verification_passed is False else 0.5
            experiment.metrics.setdefault("verification", []).append(verification_value)
            experiment.metrics.setdefault("drag", []).append(float(salience.get("drag", 0.0)))
            experiment.metrics.setdefault("cost", []).append(float(salience.get("cost", 0.0)))
            active.remaining_steps -= 1
            if active.remaining_steps <= 0:
                analysed = self.dispatcher.analyze_results(experiment)
                reports.append(
                    {
                        "name": analysed.name,
                        "conclusion": analysed.conclusion,
                        "results": analysed.results,
                    }
                )
                overrides_reverter(active.originals)
                finished.append(active)
        for entry in finished:
            self._active.remove(entry)
        return reports

    def reset(self) -> None:
        self._active.clear()


__all__ = ["ExperimentCoordinator", "ActiveExperiment"]
