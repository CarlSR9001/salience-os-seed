"""Synthetic state generators for SalienceRuntime demos."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass
class GeneratorState:
    step: int = 0
    phase: float = 0.0
    mode: str = "baseline"


class StateGenerator(ABC):
    """Protocol for synthetic state emitters."""

    @abstractmethod
    def next_state(self) -> Mapping[str, object]:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...

    def reset(self) -> None:
        pass


class BaselineGenerator(StateGenerator):
    """Deterministic low-variance generator for smoke tests."""

    def __init__(self) -> None:
        self._state = GeneratorState()

    def next_state(self) -> Mapping[str, object]:
        self._state.step += 1
        logits = np.array(
            [
                [1.5, 0.2, -0.3, -0.4],
                [1.6, 0.1, -0.2, -0.5],
            ]
        )
        return {
            "sequence_id": 0,
            "prediction": {
                "token_logits": logits,
                "steps_remaining": max(0.0, 5.0 - 0.1 * self._state.step),
                "entropy_estimate": 1.2,
            },
            "context": {
                "tokens": ["baseline", "run", str(self._state.step % 10)],
                "text": f"baseline run {self._state.step}",
            },
            "decision_proposal": {"operator": "SASS", "cot_depth": 1},
            "token_cost": 6.0,
        }

    def describe(self) -> str:
        return "Deterministic baseline sequence with steady salience."

    def reset(self) -> None:
        self._state = GeneratorState()


class SpikyGenerator(StateGenerator):
    """Generator inducing periodic uncertainty/novelty spikes."""

    def __init__(self, period: int = 8) -> None:
        self.period = period
        self._state = GeneratorState(mode="spiky")

    def next_state(self) -> Mapping[str, object]:
        self._state.step += 1
        step = self._state.step
        phase = (step % self.period) / self.period
        entropy = 1.0 + 3.0 * math.sin(2 * math.pi * phase)
        novelty_tokens = ["idea", "burst", str(step)]
        state: Dict[str, object] = {
            "sequence_id": 1,
            "prediction": {
                "token_logits": np.random.normal(0, 1, size=(2, 4)),
                "steps_remaining": max(0.0, 6.0 - 0.3 * step),
                "entropy_estimate": abs(entropy),
            },
            "context": {
                "tokens": novelty_tokens,
                "text": " ".join(novelty_tokens),
            },
            "decision_proposal": {
                "operator": "SASS_WITH_JUMP",
                "cot_depth": 3 if entropy > 2.0 else 1,
            },
            "teleport_trigger": entropy > 2.0,
            "reasoner_trigger": phase > 0.6,
            "token_cost": 7.5,
        }
        if phase > 0.75:
            state["memory_verb"] = {
                "op": "schedule_todo",
                "text": f"Investigate spike at step {step}",
                "score": entropy,
            }
        return state

    def describe(self) -> str:
        return "Periodic spikes in uncertainty/novelty with occasional memory verbs."

    def reset(self) -> None:
        self._state = GeneratorState(mode="spiky")


class DraggyGenerator(StateGenerator):
    """Generator highlighting drag increase via rapid tool/memory switches."""

    def __init__(self) -> None:
        self._state = GeneratorState(mode="draggy")
        self._active_tool = "math"
        self._tools = {
            "math": lambda payload: None,
            "retrieval": lambda payload: None,
        }

    def next_state(self) -> Mapping[str, object]:
        self._state.step += 1
        self._rotate_tool()
        prompt = f"draggy run {self._state.step}"
        return {
            "sequence_id": 2,
            "prediction": {
                "token_logits": np.random.normal(0, 0.5, size=(2, 4)),
                "steps_remaining": 4.0,
                "entropy_estimate": 1.0,
            },
            "context": {
                "tokens": prompt.split(),
                "text": prompt,
            },
            "decision_proposal": {
                "operator": "TOOL",
                "cot_depth": 1,
            },
            "token_cost": 5.0,
            "tools": self._tools,
            "tool_name": self._active_tool,
            "teleport_trigger": False,
            "drag_event": True,
        }

    def describe(self) -> str:
        return "Alternates tools rapidly to spike DRAG sensor."

    def reset(self) -> None:
        self._state = GeneratorState(mode="draggy")
        self._active_tool = "math"

    def _rotate_tool(self) -> None:
        self._active_tool = "retrieval" if self._active_tool == "math" else "math"


GENERATOR_REGISTRY: Dict[str, StateGenerator] = {
    "baseline": BaselineGenerator(),
    "spiky": SpikyGenerator(),
    "draggy": DraggyGenerator(),
}
