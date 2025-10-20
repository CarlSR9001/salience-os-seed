"""MCP bridge adapters exposing runtime facilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..core.operators import MemoryOperator
from ..core.reflection import IntrospectionInterface


@dataclass(slots=True)
class MemoryResource:
    """Expose structured memory verbs through an MCP-like interface."""

    memory_operator: MemoryOperator

    def apply(self, verb: Mapping[str, object]) -> Mapping[str, object]:
        result = self.memory_operator.execute(verb)
        return {"applied": result.applied, "metadata": result.metadata}

    def snapshot(self) -> Mapping[str, object]:
        return self.memory_operator.memory.as_runtime_mapping()


@dataclass(slots=True)
class IntrospectionResource:
    """Read-only runtime introspection exposed to MCP clients."""

    introspection: IntrospectionInterface

    def workspace_listing(self, path: str = ".") -> Sequence[Mapping[str, object]]:
        return self.introspection.get_workspace_listing(path)

    def yearning_state(self) -> Mapping[str, Mapping[str, float]]:
        return self.introspection.get_yearning_state(refresh=True)

    def controller_dynamics(self) -> Mapping[str, object]:
        return self.introspection.controller_snapshot()


@dataclass(slots=True)
class SessionOrchestrator:
    """High-level commands suitable for MCP invocation."""

    runtime_stepper: callable
    reset_runtime: callable

    def start_epoch(self, steps: int = 1) -> Iterable[Mapping[str, object]]:
        for _ in range(max(1, steps)):
            metrics = self.runtime_stepper()
            yield {
                "step": metrics.step,
                "meta": metrics.meta_report,
                "verification": metrics.verification_passed,
            }

    def reset(self) -> None:
        self.reset_runtime()


__all__ = [
    "MemoryResource",
    "IntrospectionResource",
    "SessionOrchestrator",
]
