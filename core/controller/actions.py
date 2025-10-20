"""Controller action enumerations and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping


class ControllerOperator(Enum):
    """High-level operator families."""

    SASS = auto()
    SASS_WITH_JUMP = auto()
    MEMORY_OP = auto()
    TOOL = auto()
    VERIFY = auto()
    REFLECT = auto()


class ControllerPatch(Enum):
    """Skill patch selections applied per step."""

    NONE = auto()
    MATH = auto()
    RETRIEVAL = auto()
    PLAN = auto()


@dataclass(frozen=True)
class ControllerAction:
    """Action tuple specifying depth/operator/patch choices."""

    cot_depth: int
    operator: ControllerOperator
    patch: ControllerPatch


@dataclass(frozen=True)
class ControllerDecision:
    """Decisions returned by the controller at runtime."""

    action: ControllerAction
    score: float
    salience_mapping: Mapping[str, float]
    cooldown_steps: int
    hysteresis_delta: float
