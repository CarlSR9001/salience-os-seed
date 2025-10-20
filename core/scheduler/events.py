"""Event definitions for the Event-Driven Scheduler (EDN).

Events are declared declaratively via `EventDefinition`. Each event specifies:
- `name`: human-readable identifier.
- `sensor_key`: salience vector entry to monitor.
- `threshold`: value at which the event should fire.
- `comparison`: one of `gt`, `lt`, `ge`, `le`.
- `cooldown`: minimum steps between consecutive firings.
- `hysteresis`: margin that must be exceeded to reset the cooldown and avoid
  repeated toggling just around the threshold.

`EventRegistry` holds all definitions and exposes helper methods to evaluate
incoming salience readings and determine which events are active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping


@dataclass(frozen=True)
class EventDefinition:
    """Declarative specification of an EDN event."""

    name: str
    sensor_key: str
    threshold: float
    comparison: str = "gt"
    cooldown: int = 4
    hysteresis: float = 0.2


@dataclass
class EventTrigger:
    """Concrete trigger state for an event."""

    definition: EventDefinition
    cooldown_remaining: int = 0
    last_value: float = 0.0
    fired_last_tick: bool = False

    def evaluate(self, salience: Mapping[str, float]) -> bool:
        value = float(salience.get(self.definition.sensor_key, 0.0))
        self.last_value = value
        self.fired_last_tick = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False
        comparator = _COMPARATORS[self.definition.comparison]
        if comparator(value, self.definition.threshold):
            self.cooldown_remaining = self.definition.cooldown
            self.fired_last_tick = True
            return True
        if comparator(value, self.definition.threshold - self.definition.hysteresis):
            # still above hysteresis band; hold cooldown at zero but do not fire
            return False
        return False


class EventRegistry:
    """Holds event definitions and manages trigger lifecycle."""

    def __init__(self, definitions: Iterable[EventDefinition]) -> None:
        self._definitions = list(definitions)
        self._triggers: Dict[str, EventTrigger] = {
            definition.name: EventTrigger(definition)
            for definition in self._definitions
        }
        if not self._definitions:
            raise ValueError("EventRegistry requires at least one event definition")

    def evaluate(self, salience: Mapping[str, float]) -> List[EventTrigger]:
        fired: List[EventTrigger] = []
        for trigger in self._triggers.values():
            if trigger.evaluate(salience):
                fired.append(trigger)
        return fired

    def snapshot(self) -> Mapping[str, Mapping[str, float]]:
        return {
            name: {
                "cooldown_remaining": float(trigger.cooldown_remaining),
                "last_value": trigger.last_value,
                "threshold": trigger.definition.threshold,
            }
            for name, trigger in self._triggers.items()
        }

    def reset(self) -> None:
        for trigger in self._triggers.values():
            trigger.cooldown_remaining = 0
            trigger.last_value = 0.0
            trigger.fired_last_tick = False


def default_registry() -> EventRegistry:
    """Create the canonical event registry described in the spec."""

    definitions = [
        EventDefinition(name="novelty_peak", sensor_key="novelty", threshold=1.2, comparison="gt", cooldown=3),
        EventDefinition(name="uncertainty_spike", sensor_key="uncertainty", threshold=1.5, comparison="gt", cooldown=2),
        EventDefinition(name="key_gate", sensor_key="progress", threshold=1.0, comparison="gt", cooldown=1),
        EventDefinition(name="drag_surge", sensor_key="drag", threshold=1.0, comparison="gt", cooldown=4),
    ]
    return EventRegistry(definitions)


def _gt(value: float, threshold: float) -> bool:
    return value > threshold


def _ge(value: float, threshold: float) -> bool:
    return value >= threshold


def _lt(value: float, threshold: float) -> bool:
    return value < threshold


def _le(value: float, threshold: float) -> bool:
    return value <= threshold


_COMPARATORS = {
    "gt": _gt,
    "ge": _ge,
    "lt": _lt,
    "le": _le,
}
