"""Event-driven scheduler implementation.

The scheduler checks salience-triggered events before allowing expensive
operators to execute. It supports:
- Event registry evaluation.
- Cooldown tracking to avoid firing events too frequently.
- Budget-aware gating: even if an event fires, the scheduler can defer execution
  when the remaining compute budget is critically low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

from .events import EventRegistry, EventTrigger, default_registry


@dataclass
class SchedulerConfig:
    """Configuration for the event-driven scheduler."""

    min_budget_ratio: float = 0.05
    cooldown_bias: float = 0.1


@dataclass
class SchedulerState:
    """Mutable runtime state for the scheduler."""

    last_fired_events: List[str] = field(default_factory=list)
    budget_ratio: float = 1.0


class EventDrivenScheduler:
    """Evaluate salience events and gate operator execution."""

    def __init__(
        self,
        registry: EventRegistry | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.config = config or SchedulerConfig()
        self.state = SchedulerState()

    def should_fire(
        self,
        salience: Mapping[str, float],
        decision_operator: str,
        budget_left: float,
        budget_total: float,
    ) -> bool:
        """Determine whether to execute the operator on this step."""

        budget_ratio = budget_left / max(budget_total, 1e-6)
        self.state.budget_ratio = budget_ratio
        if budget_ratio < self.config.min_budget_ratio:
            self.state.last_fired_events = []
            return decision_operator.upper() == "VERIFY"

        triggers = self.registry.evaluate(salience)
        self.state.last_fired_events = [trigger.definition.name for trigger in triggers]
        if not triggers:
            return decision_operator.upper() == "SASS"
        # Weighted evaluation to decide whether the event should actually gate execution
        votes = sum(self._event_vote(trigger, salience) for trigger in triggers)
        return votes > 0.5

    def snapshot(self) -> Mapping[str, object]:
        return {
            "events": self.state.last_fired_events,
            "budget_ratio": self.state.budget_ratio,
            "registry": self.registry.snapshot(),
        }

    def reset(self) -> None:
        self.registry.reset()
        self.state = SchedulerState()

    def _event_vote(self, trigger: EventTrigger, salience: Mapping[str, float]) -> float:
        if trigger.fired_last_tick:
            return 1.0
        value = float(salience.get(trigger.definition.sensor_key, 0.0))
        threshold = trigger.definition.threshold
        delta = max(0.0, value - threshold)
        cooldown_factor = 1.0 / (1.0 + trigger.cooldown_remaining + self.config.cooldown_bias)
        return delta * cooldown_factor
