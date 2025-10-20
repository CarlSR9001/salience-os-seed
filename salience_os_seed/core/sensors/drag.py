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

import numpy as np

from .base import MedianMADNormalizer, Sensor


class DragSensor(Sensor):
    """Measure prompt/tool thrash to discourage oscillations."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        prompt_weight: float = 0.6,
        switch_weight: float = 0.4,
        novelty_refresh_threshold: float = 0.55,
        anchor_confidence: float = 0.7,
    ) -> None:
        super().__init__(name="drag", domain="drag", normaliser=normaliser)
        self._prompt_weight = float(prompt_weight)
        self._switch_weight = float(switch_weight)
        self._novelty_refresh_threshold = float(np.clip(novelty_refresh_threshold, 0.0, 1.0))
        self._anchor_confidence = float(np.clip(anchor_confidence, 0.0, 1.0))
        self._distance_source: str = "exact"

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        stats = memory.setdefault("drag_stats", {  # type: ignore[attr-defined]
            "last_prompt": "",
            "switch_count": 0.0,
            "anchor_prompt": "",
            "cached_distance": 0.0,
            "distance_source": "exact",
        })  # type: ignore[assignment]
        prompt = _extract_prompt(state)
        novelty_hint = float(meta.get("novelty", meta.get("novelty_score", 0.0)) or 0.0)
        novelty_hint = float(np.clip(novelty_hint, 0.0, 1.0))
        confidence_hint = float(meta.get("confidence", 0.0) or 0.0)
        roi_hint = float(meta.get("roi", 0.0) or 0.0)
        anchor_override = meta.get("drag_anchor")
        if isinstance(anchor_override, str) and anchor_override.strip():
            stats["anchor_prompt"] = anchor_override
        elif confidence_hint >= self._anchor_confidence or roi_hint >= self._anchor_confidence:
            stats["anchor_prompt"] = prompt
        elif not stats.get("anchor_prompt"):
            stats["anchor_prompt"] = prompt

        last_prompt = stats.get("last_prompt", "")
        anchor_prompt = stats.get("anchor_prompt", last_prompt)

        if prompt == last_prompt:
            prompt_distance = float(stats.get("cached_distance", 0.0))
            distance_source = stats.get("distance_source", "exact")
        else:
            if novelty_hint >= self._novelty_refresh_threshold:
                prompt_distance = _normalised_levenshtein(anchor_prompt, prompt)
                distance_source = "exact"
            else:
                approx_anchor = _token_delta_ratio(anchor_prompt, prompt)
                approx_last = _token_delta_ratio(last_prompt, prompt)
                prompt_distance = max(approx_anchor, approx_last, float(stats.get("cached_distance", 0.0)))
                distance_source = "approx"

        stats["cached_distance"] = prompt_distance
        stats["distance_source"] = distance_source
        switch_count = float(stats.get("switch_count", 0.0))
        drag_score = self._prompt_weight * prompt_distance + self._switch_weight * switch_count
        stats["last_prompt"] = prompt
        stats["switch_count"] = 0.0
        self._distance_source = distance_source
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
            "distance_source_exact": 1.0 if self._distance_source == "exact" else 0.0,
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


def _token_delta_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        length_penalty = abs(len(a) - len(b)) / max(len(a), len(b), 1)
        return float(np.clip(length_penalty, 0.0, 1.0))
    symmetric = tokens_a.symmetric_difference(tokens_b)
    union_size = max(len(tokens_a.union(tokens_b)), 1)
    lexical = len(symmetric) / union_size
    length_penalty = abs(len(a) - len(b)) / max(len(a), len(b), 1)
    return float(np.clip(0.6 * lexical + 0.4 * length_penalty, 0.0, 1.0))


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
