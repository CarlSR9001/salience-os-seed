"""Runtime driver wrapping SalienceRuntime for interactive demos."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Mapping, MutableMapping, Optional

from ..core.meta import MetaState
from ..core.memory import StructuredMemory
from ..core.operators import MemoryOperator
from ..core.sensors import SensorBank
from .orchestrator import RuntimeConfig, RuntimeMetrics, SalienceRuntime
from .state_gen import GENERATOR_REGISTRY, StateGenerator


@dataclass
class DriverSnapshot:
    """Snapshot exported to the UI for rendering."""

    meta_report: str
    metrics: RuntimeMetrics
    last_metrics: Iterable[RuntimeMetrics]
    generator_name: str
    generator_description: str
    memory_snapshot: Mapping[str, object]


@dataclass
class RuntimeDriver:
    """High-level controller coordinating runtime + generator."""

    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    history_size: int = 32

    def __post_init__(self) -> None:
        self.runtime = SalienceRuntime(self.config)
        self.generators: MutableMapping[str, StateGenerator] = GENERATOR_REGISTRY
        self.generator_key = "baseline"
        self.state_generator = self.generators[self.generator_key]
        self.history: Deque[RuntimeMetrics] = deque(maxlen=self.history_size)

    def set_generator(self, key: str) -> None:
        if key not in self.generators:
            raise KeyError(f"Unknown generator '{key}'")
        self.generator_key = key
        self.state_generator = self.generators[key]
        self.state_generator.reset()

    def available_generators(self) -> Mapping[str, str]:
        return {key: gen.describe() for key, gen in self.generators.items()}

    def step(self) -> DriverSnapshot:
        state = self.state_generator.next_state()
        metrics = self.runtime.run_step(state)
        self.history.append(metrics)
        return DriverSnapshot(
            meta_report=metrics.meta_report,
            metrics=metrics,
            last_metrics=tuple(self.history),
            generator_name=self.generator_key,
            generator_description=self.state_generator.describe(),
            memory_snapshot=self.runtime.memory.serialize(),
        )

    def reset(self) -> None:
        self.runtime = SalienceRuntime(self.config)
        self.state_generator.reset()
        self.history.clear()

    def inject_memory(self, verb: Mapping[str, object]) -> None:
        memory_operator = MemoryOperator(self.runtime.memory)
        memory_operator.execute(verb)

    def snapshot(self) -> DriverSnapshot:
        if not self.history:
            return self.step()
        metrics = self.history[-1]
        return DriverSnapshot(
            meta_report=metrics.meta_report,
            metrics=metrics,
            last_metrics=tuple(self.history),
            generator_name=self.generator_key,
            generator_description=self.state_generator.describe(),
            memory_snapshot=self.runtime.memory.serialize(),
        )
