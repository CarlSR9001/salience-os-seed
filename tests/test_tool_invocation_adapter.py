"""Tests for ToolInvocationAdapter resolution paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import importlib
import sys
import types
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "salience_os_seed"


def _ensure_namespace(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    else:
        module.__path__ = [str(path)]
    return module


salience_pkg = _ensure_namespace("salience_os_seed", _PACKAGE_ROOT)
core_pkg = _ensure_namespace("salience_os_seed.core", _PACKAGE_ROOT / "core")
runtime_pkg = _ensure_namespace("salience_os_seed.runtime", _PACKAGE_ROOT / "runtime")


class ControllerOperator(Enum):
    SASS = auto()
    SASS_WITH_JUMP = auto()
    MEMORY_OP = auto()
    TOOL = auto()
    VERIFY = auto()
    REFLECT = auto()


class ControllerPatch(Enum):
    NONE = auto()
    MATH = auto()
    RETRIEVAL = auto()
    PLAN = auto()


@dataclass(frozen=True)
class ControllerAction:
    cot_depth: int
    operator: ControllerOperator
    patch: ControllerPatch


@dataclass(frozen=True)
class ControllerDecision:
    action: ControllerAction
    score: float = 0.0
    salience_mapping: dict[str, float] | None = None
    cooldown_steps: int = 0
    hysteresis_delta: float = 0.0


class BanditTrainer:  # pragma: no cover - stub satisfying import
    def update(self, action: ControllerAction, reward: float) -> None:  # pragma: no cover
        pass


controller_module = types.ModuleType("salience_os_seed.core.controller")
controller_module.BanditTrainer = BanditTrainer
controller_module.ControllerAction = ControllerAction
controller_module.ControllerDecision = ControllerDecision
controller_module.ControllerOperator = ControllerOperator
controller_module.ControllerPatch = ControllerPatch
sys.modules["salience_os_seed.core.controller"] = controller_module
core_pkg.controller = controller_module


memory_module = types.ModuleType("salience_os_seed.core.memory")
memory_module.StructuredMemory = type("StructuredMemory", (), {})
sys.modules["salience_os_seed.core.memory"] = memory_module
core_pkg.memory = memory_module


operators_module = types.ModuleType("salience_os_seed.core.operators")
operators_module.GraphReasoner = type("GraphReasoner", (), {})
operators_module.MemoryOperator = type("MemoryOperator", (), {})
operators_module.SparseJumpTeleporter = type("SparseJumpTeleporter", (), {})
operators_module.VerifierSuite = type("VerifierSuite", (), {})
operators_module.SASSCore = type("SASSCore", (), {})
sys.modules["salience_os_seed.core.operators"] = operators_module
core_pkg.operators = operators_module


reflection_module = types.ModuleType("salience_os_seed.core.reflection")
reflection_module.IntrospectionInterface = type("IntrospectionInterface", (), {})
sys.modules["salience_os_seed.core.reflection"] = reflection_module
core_pkg.reflection = reflection_module


if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(Tensor=type("Tensor", (), {}))


ToolInvocationAdapter = importlib.import_module(
    "salience_os_seed.runtime.action_executor"
).ToolInvocationAdapter


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, state: dict[str, Any]) -> None:
        self.calls.append(state.get("payload"))


def _tool_action(operator: ControllerOperator, patch: ControllerPatch) -> ControllerAction:
    return ControllerAction(cot_depth=1, operator=operator, patch=patch)


def test_adapter_prefers_explicit_tool_name_from_state() -> None:
    recorder = _Recorder()
    adapter = ToolInvocationAdapter()
    action = _tool_action(ControllerOperator.TOOL, ControllerPatch.MATH)
    state = {
        "tool_name": "helper",
        "tools": {"helper": recorder},
        "payload": "from-state",
    }

    assert adapter.invoke(action, state) is True
    assert recorder.calls == ["from-state"]


def test_adapter_uses_operator_mapping_when_no_explicit_name() -> None:
    recorder = _Recorder()
    adapter = ToolInvocationAdapter(
        runtime_tools={"helper": recorder},
        operator_tool_map={ControllerOperator.TOOL: {ControllerPatch.MATH: "helper"}},
    )
    action = _tool_action(ControllerOperator.TOOL, ControllerPatch.MATH)

    assert adapter.invoke(action, {"payload": "runtime"}) is True
    assert recorder.calls == ["runtime"]


def test_adapter_falls_back_to_known_patch_name() -> None:
    recorder = _Recorder()
    adapter = ToolInvocationAdapter(runtime_tools={"plan": recorder})
    action = _tool_action(ControllerOperator.TOOL, ControllerPatch.PLAN)

    assert adapter.invoke(action, {"payload": "fallback"}) is True
    assert recorder.calls == ["fallback"]


def test_adapter_requires_known_tool_for_patch_fallback() -> None:
    adapter = ToolInvocationAdapter()
    action = _tool_action(ControllerOperator.TOOL, ControllerPatch.MATH)

    assert adapter.invoke(action, {}) is False


def test_adapter_ignores_none_patch_when_unresolved() -> None:
    recorder = _Recorder()
    adapter = ToolInvocationAdapter(runtime_tools={"none": recorder})
    action = _tool_action(ControllerOperator.TOOL, ControllerPatch.NONE)

    assert adapter.invoke(action, {}) is False
    assert recorder.calls == []
