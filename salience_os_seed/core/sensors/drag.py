"""DRAG sensor quantifying friction from memory/tool churn.

DRAG rises when the system keeps editing prompts, switching tools, or thrashing
structured memory. We derive the score from:
- Edit distance between the current prompt slice and the last stabilised prompt.
- Count of memory/tool switches since the previous decision frame.

The signal is essential for discouraging the controller from oscillating between
strategies; high DRAG penalises actions through the controller score.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping

from .base import MedianMADNormalizer, Sensor


class DragSensor(Sensor):
    """Measure prompt/tool thrash to discourage oscillations."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        prompt_weight: float = 0.6,
        switch_weight: float = 0.4,
    ) -> None:
        super().__init__(name="drag", domain="drag", normaliser=normaliser)
        self._prompt_weight = float(prompt_weight)
        self._switch_weight = float(switch_weight)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        stats = memory.setdefault("drag_stats", {  # type: ignore[attr-defined]
            "last_prompt": "",
            "switch_count": 0.0,
        })  # type: ignore[assignment]
        prompt = _extract_prompt(state)
        prompt_distance = _normalised_levenshtein(stats.get("last_prompt", ""), prompt)
        switch_count = float(stats.get("switch_count", 0.0))
        drag_score = self._prompt_weight * prompt_distance + self._switch_weight * switch_count
        stats["last_prompt"] = prompt
        stats["switch_count"] = 0.0
        return drag_score

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        return {
            "raw_drag": raw_value,
            "normalised": norm_value,
        }


def record_tool_switch(memory: MutableMapping[str, object]) -> None:
    """Increment tool/memory switch counters for the next tick.

    Runtime modules call this helper each time they switch tool contexts or edit
    structured memory in a way that should increase DRAG.
    """

    stats = memory.setdefault("drag_stats", {"switch_count": 0.0})  # type: ignore[attr-defined]
    stats["switch_count"] = stats.get("switch_count", 0.0) + 1.0


def _extract_prompt(state: Mapping[str, object]) -> str:
    prompt = state.get("context")
    if isinstance(prompt, Mapping):
        text = prompt.get("text")
        if isinstance(text, str):
            return text
    return str(state.get("prompt", ""))


def _normalised_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    distance = _levenshtein(a, b)
    return distance / max(len(a), len(b), 1)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
