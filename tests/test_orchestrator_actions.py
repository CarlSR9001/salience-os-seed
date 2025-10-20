"""Regression coverage for runtime action execution branches."""

from __future__ import annotations

import math
import statistics
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from typing import Mapping


def _install_package_alias() -> None:
    if "salience_os_seed" in sys.modules:
        return
    module = types.ModuleType("salience_os_seed")
    module.__path__ = [str(Path(__file__).resolve().parents[1] / "salience_os_seed")]
    sys.modules["salience_os_seed"] = module


def _install_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return

    class _SimpleArray:
        def __init__(self, data):
            self._data = [float(x) for x in data]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, index):
            return self._data[index]

        @property
        def size(self):  # pragma: no cover - compatibility shim
            return len(self._data)

        def __setitem__(self, index, value):
            self._data[index] = float(value)

        def __add__(self, other):
            return _SimpleArray(a + b for a, b in zip(self._data, _coerce_iter(other, len(self))))

        __radd__ = __add__

        def __sub__(self, other):
            return _SimpleArray(a - b for a, b in zip(self._data, _coerce_iter(other, len(self))))

        def __rsub__(self, other):
            return _SimpleArray(b - a for a, b in zip(self._data, _coerce_iter(other, len(self))))

        def __mul__(self, other):
            return _SimpleArray(a * b for a, b in zip(self._data, _coerce_iter(other, len(self))))

        __rmul__ = __mul__

        def __truediv__(self, other):
            return _SimpleArray(a / b for a, b in zip(self._data, _coerce_iter(other, len(self))))

        def __rtruediv__(self, other):
            return _SimpleArray(b / a for a, b in zip(self._data, _coerce_iter(other, len(self))))

        def __itruediv__(self, other):
            other_iter = _coerce_iter(other, len(self))
            for idx, value in enumerate(other_iter):
                self._data[idx] /= value
            return self

        def tolist(self):  # pragma: no cover - convenience helper
            return list(self._data)

    def _coerce_iter(value, length):
        if isinstance(value, _SimpleArray):
            return value
        try:
            iterator = iter(value)  # type: ignore[arg-type]
            return _SimpleArray(iterator)
        except TypeError:
            return _SimpleArray([value] * length)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = _SimpleArray
    numpy_stub.float32 = float
    numpy_stub.float64 = float
    numpy_stub.floating = float
    numpy_stub.integer = int

    def _as_array(data):
        if isinstance(data, _SimpleArray):
            return data
        try:
            return _SimpleArray(data)
        except TypeError:
            return _SimpleArray([data])

    numpy_stub.asarray = lambda data, dtype=None: _as_array(data)
    numpy_stub.fromiter = lambda iterable, dtype=None: _SimpleArray(list(iterable))
    numpy_stub.zeros = lambda shape, dtype=None: _SimpleArray([0.0] * (shape[0] if isinstance(shape, tuple) else int(shape)))

    def _median(arr):
        values = list(arr)
        return statistics.median(values) if values else 0.0

    def _mean(seq):
        values = list(seq)
        return (sum(values) / len(values)) if values else 0.0

    numpy_stub.median = _median
    numpy_stub.mean = _mean
    numpy_stub.abs = lambda value: _SimpleArray(abs(x) for x in value) if isinstance(value, _SimpleArray) else abs(value)
    numpy_stub.tanh = math.tanh

    def _std(seq):
        values = list(seq)
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    def _histogram(seq, bins=10, range_=None):
        values = list(seq)
        if not values:
            return _SimpleArray([0] * bins), _SimpleArray([0.0] * (bins + 1))
        if range_ is None:
            min_value = min(values)
            max_value = max(values)
        else:
            min_value, max_value = map(float, range_)
        if math.isclose(max_value, min_value):
            max_value = min_value + 1.0
        width = (max_value - min_value) / bins
        edges = [min_value + idx * width for idx in range(bins + 1)]
        counts = [0] * bins
        for value in values:
            idx = int((value - min_value) / width)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        return _SimpleArray(counts), _SimpleArray(edges)

    def _clip(value, min_value, max_value):
        if isinstance(value, _SimpleArray):
            return _SimpleArray(min(max(x, min_value), max_value) for x in value)
        return min(max(value, min_value), max_value)

    numpy_stub.clip = _clip
    numpy_stub.dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    numpy_stub.std = _std
    numpy_stub.min = lambda seq: min(seq) if seq else 0.0
    numpy_stub.max = lambda seq: max(seq) if seq else 0.0
    numpy_stub.histogram = _histogram
    numpy_stub.linalg = types.SimpleNamespace(norm=lambda arr: math.sqrt(sum(x * x for x in arr)))

    class _DummyRandom:
        def __init__(self, seed=None):
            self.seed = seed

        def normal(self, loc=0.0, scale=1.0, size=None):
            if size is None:
                return float(loc)
            if isinstance(size, tuple):
                count = 1
                for dim in size:
                    count *= int(dim)
                return _SimpleArray([float(loc)] * count)
            return _SimpleArray([float(loc)] * int(size))

    numpy_stub.random = types.SimpleNamespace(default_rng=lambda seed=None: _DummyRandom(seed))
    sys.modules["numpy"] = numpy_stub


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    class _Tensor:
        def __init__(self, data=None):
            self._data = data

        def detach(self):
            return self

        def cpu(self):  # pragma: no cover - compatibility hook
            return self

        def numpy(self):  # pragma: no cover - compatibility hook
            array = sys.modules["numpy"].asarray(self._data if self._data is not None else [])
            return array

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _Tensor
    torch_stub.tensor = lambda data=None, dtype=None: _Tensor(data)
    sys.modules["torch"] = torch_stub


def _install_operator_stubs() -> None:
    sass_name = "salience_os_seed.core.operators.sass"
    if sass_name not in sys.modules:
        sass_module = types.ModuleType(sass_name)

        class SASSConfig:
            pass

        class SASSCore:
            def __init__(self, config):
                self.config = config

            def __call__(self, hidden_states, layer_states=None, hyper_deltas=None):
                return hidden_states, list(layer_states or [])

        sass_module.SASSConfig = SASSConfig
        sass_module.SASSCore = SASSCore
        sys.modules[sass_name] = sass_module

    sparse_name = "salience_os_seed.core.operators.sparse_jump"
    if sparse_name not in sys.modules:
        sparse_module = types.ModuleType(sparse_name)

        class SparseJumpTeleporter:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, token_states, sequence_id=0, trigger=False):
                return token_states

        sparse_module.SparseJumpTeleporter = SparseJumpTeleporter
        sys.modules[sparse_name] = sparse_module

    graph_name = "salience_os_seed.core.operators.graph_reasoner"
    if graph_name not in sys.modules:
        graph_module = types.ModuleType(graph_name)

        class GraphReasonerConfig:
            pass

        class GraphReasoner:
            def __init__(self, config):
                self.config = config

            def __call__(self, output, memory, entity_hints=()):
                return output, {}

        graph_module.GraphReasonerConfig = GraphReasonerConfig
        graph_module.GraphReasoner = GraphReasoner
        sys.modules[graph_name] = graph_module

    verifier_name = "salience_os_seed.core.operators.verifier"
    if verifier_name not in sys.modules:
        verifier_module = types.ModuleType(verifier_name)

        class VerificationOutcome:
            def __init__(self, passed: bool = True, evidence: str | None = None):
                self.passed = passed
                self.evidence = evidence or ""

        class VerifierSuite:
            def run(self, context):
                return VerificationOutcome(passed=True)

        verifier_module.VerificationOutcome = VerificationOutcome
        verifier_module.VerifierSuite = VerifierSuite
        sys.modules[verifier_name] = verifier_module


def _install_meta_stubs() -> None:
    meta_name = "salience_os_seed.core.meta"
    if meta_name in sys.modules:
        return

    meta_module = types.ModuleType(meta_name)
    meta_module.__path__ = []

    from dataclasses import dataclass

    @dataclass
    class MetaStateConfig:
        vector_dim: int = 8
        salience_dim: int = 6
        history_window: int = 64
        confidence_index: int = 0
        difficulty_index: int = 1
        blindspot_index: int = 2
        roi_index: int = 3

    class MetaState:
        def __init__(self, config: MetaStateConfig) -> None:
            self.config = config
            self.vector = [0.0] * config.vector_dim
            self.history: list[dict[str, float]] = []

        def update(
            self,
            salience_vector: Mapping[str, float],
            verification_passed: bool | None,
            budget_left: float,
            cooldown_active: bool,
        ) -> list[float]:
            return list(self.vector)

        def snapshot(self) -> Mapping[str, float]:
            cfg = self.config
            return {
                "confidence": float(self.vector[cfg.confidence_index]),
                "difficulty": float(self.vector[cfg.difficulty_index]),
                "blind_spot": float(self.vector[cfg.blindspot_index]),
                "roi": float(self.vector[cfg.roi_index]),
            }

    class EpisodeRecord:
        def __init__(self, episode_id: str) -> None:
            self.episode_id = episode_id

    class EpisodicStore:
        def __init__(self, *_args, **_kwargs) -> None:
            self._episodes: list[str] = []

        def record_episode(self, episode) -> EpisodeRecord:
            episode_id = f"episode-{len(self._episodes)}"
            self._episodes.append(episode_id)
            return EpisodeRecord(episode_id)

    def build_episode(**kwargs):  # pragma: no cover - simple stub
        return kwargs

    def render_self_report(*_args, **_kwargs) -> str:  # pragma: no cover - simple stub
        return ""

    meta_module.MetaStateConfig = MetaStateConfig
    meta_module.MetaState = MetaState
    meta_module.EpisodicStore = EpisodicStore
    meta_module.build_episode = build_episode
    meta_module.render_self_report = render_self_report
    sys.modules[meta_name] = meta_module

    state_module = types.ModuleType(f"{meta_name}.state")
    state_module.MetaStateConfig = MetaStateConfig
    state_module.MetaState = MetaState
    sys.modules[f"{meta_name}.state"] = state_module


_install_package_alias()
_install_numpy_stub()
_install_torch_stub()
_install_operator_stubs()
_install_meta_stubs()

from salience_os_seed.core.controller import (
    ControllerAction,
    ControllerDecision,
    ControllerOperator,
    ControllerPatch,
)
from salience_os_seed.runtime.orchestrator import SalienceRuntime


DEFAULT_SALIENCE = {
    "progress": 0.8,
    "uncertainty": 0.3,
    "novelty": 0.4,
    "alignment": 0.5,
    "cost": 0.1,
    "drag": 0.2,
    "retention": 0.8,
    "roi": 0.9,
}


class _DummySalienceVector:
    def __init__(self, mapping: Mapping[str, float]) -> None:
        self._mapping = dict(mapping)
        self.readings = [types.SimpleNamespace(name=key, raw=value) for key, value in self._mapping.items()]

    def as_mapping(self) -> Mapping[str, float]:
        return dict(self._mapping)


def _configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    operator: ControllerOperator,
    *,
    salience_override: dict[str, float] | None = None,
) -> tuple[SalienceRuntime, ControllerDecision, dict[str, float]]:
    runtime = SalienceRuntime()
    salience = dict(DEFAULT_SALIENCE)
    if salience_override:
        salience.update(salience_override)

    action = ControllerAction(cot_depth=1, operator=operator, patch=ControllerPatch.NONE)
    decision = ControllerDecision(
        action=action,
        score=0.0,
        salience_mapping=dict(salience),
        cooldown_steps=0,
        hysteresis_delta=0.0,
    )

    def fake_choose(salience_map, meta_snapshot, budget_left):  # pragma: no cover - simple passthrough
        runtime.controller.state.last_action = decision.action
        runtime.controller.state.cooldown_remaining = 0
        return decision

    monkeypatch.setattr(runtime.controller, "choose", fake_choose)
    monkeypatch.setattr(runtime.scheduler, "should_fire", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime, "_prepare_auction_bids", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_maybe_generate_ideas", lambda *args, **kwargs: 0)

    def fake_tick(state, memory, meta):  # pragma: no cover - deterministic test stub
        return _DummySalienceVector(salience)

    monkeypatch.setattr(runtime.sensor_bank, "tick", fake_tick)
    return runtime, decision, salience


def _bandit_bucket(runtime: SalienceRuntime, action: ControllerAction) -> dict[str, float]:
    key = runtime.bandit_trainer._key(action)
    return runtime.bandit_trainer.weight_store.get(key, {})


def test_memory_operator_updates_memory_and_bandit(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, decision, _ = _configure_runtime(monkeypatch, ControllerOperator.MEMORY_OP)
    verb = {"op": "schedule_todo", "text": "file regression test"}
    runtime.run_step({"memory_verb": verb, "token_cost": 0.0})

    assert runtime.memory.todos.count() == 1
    bucket = _bandit_bucket(runtime, decision.action)
    assert bucket.get("count", 0.0) >= 1.0


def test_tool_operator_invokes_tool_and_trains(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, decision, _ = _configure_runtime(monkeypatch, ControllerOperator.TOOL)
    invocations: list[str] = []

    def helper_tool(state):  # pragma: no cover - invoked via runtime
        invocations.append(state.get("payload", ""))

    state = {
        "tools": {"helper": helper_tool},
        "tool_name": "helper",
        "payload": "ping",
        "token_cost": 0.0,
    }
    runtime.run_step(state)

    assert invocations == ["ping"]
    bucket = _bandit_bucket(runtime, decision.action)
    assert bucket.get("count", 0.0) >= 1.0


def test_reflect_operator_commits_trace_and_trains(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, decision, _ = _configure_runtime(
        monkeypatch,
        ControllerOperator.REFLECT,
        salience_override={"novelty": 0.7, "uncertainty": 0.6},
    )
    runtime.run_step({"context_snippet": "capture learnings", "token_cost": 0.0})

    assert len(runtime.scratchpad.trace_history) == 1
    bucket = _bandit_bucket(runtime, decision.action)
    assert bucket.get("count", 0.0) >= 1.0


def test_patched_decision_updates_controller_state(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _, _ = _configure_runtime(monkeypatch, ControllerOperator.SASS)
    executed: list[ControllerOperator] = []

    def fake_execute(self, decision, state, salience):  # pragma: no cover - deterministic stub
        executed.append(decision.action.operator)
        return None

    monkeypatch.setattr(type(runtime.action_executor), "execute", fake_execute)

    metrics = runtime.run_step({"token_cost": 0.0})

    assert metrics.decision.action.operator is ControllerOperator.REFLECT
    assert executed == [ControllerOperator.REFLECT]
    assert runtime.controller.state.last_action == metrics.decision.action
    assert runtime.controller.state.last_score == metrics.decision.score
    assert runtime.controller.state.cooldown_remaining == metrics.decision.cooldown_steps
