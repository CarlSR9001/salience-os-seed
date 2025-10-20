"""Salience-aware gradient aggregation tuned for the seed runtime.

The ``AdaptiveGradientFlow`` class collects outcome signals from runtime steps and
produces smoothed gradient estimates for novelty, retention, payoff, and cost.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Optional


@dataclass
class FlowSignal:
    task: str
    reward: float
    penalty: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def net(self) -> float:
        return self.reward - self.penalty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "reward": self.reward,
            "penalty": self.penalty,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FlowSignal":
        return cls(
            task=str(payload.get("task", "")),
            reward=float(payload.get("reward", 0.0)),
            penalty=float(payload.get("penalty", 0.0)),
            components=dict(payload.get("components", {})),
            metadata=dict(payload.get("metadata", {})),
            timestamp=float(payload.get("timestamp", time.time())),
        )


@dataclass
class FlowEstimate:
    task: str
    gradients: Dict[str, float]
    bias: Dict[str, float]
    support: int
    avg_reward: float
    avg_penalty: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def net(self) -> float:
        return self.avg_reward - self.avg_penalty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "gradients": dict(self.gradients),
            "bias": dict(self.bias),
            "support": self.support,
            "avg_reward": self.avg_reward,
            "avg_penalty": self.avg_penalty,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FlowEstimate":
        return cls(
            task=str(payload.get("task", "")),
            gradients=dict(payload.get("gradients", {})),
            bias=dict(payload.get("bias", {})),
            support=int(payload.get("support", 0)),
            avg_reward=float(payload.get("avg_reward", 0.0)),
            avg_penalty=float(payload.get("avg_penalty", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


class AdaptiveGradientFlow:
    def __init__(self, *, window: int = 128, smoothing: float = 0.4) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.smoothing = max(0.0, min(1.0, smoothing))
        self._signals: Dict[str, Deque[FlowSignal]] = defaultdict(lambda: deque(maxlen=self.window))
        self._history: Dict[str, Deque[FlowEstimate]] = defaultdict(deque)

    def record(self, signal: FlowSignal) -> None:
        bucket = self._signals[signal.task]
        bucket.append(signal)

    def has_signals(self, task: str) -> bool:
        signals = self._signals.get(task)
        return bool(signals)

    def estimate(self, task: str) -> Optional[FlowEstimate]:
        signals = self._signals.get(task)
        if not signals:
            return None
        return self._summarize(task, signals)

    def drain(self, task: str) -> Optional[FlowEstimate]:
        signals = self._signals.get(task)
        if not signals:
            return None
        estimate = self._summarize(task, signals)
        history = self._history[task]
        history.append(estimate)
        if len(history) > 64:
            history.popleft()
        signals.clear()
        return estimate

    def history(self, task: Optional[str] = None) -> Dict[str, Iterable[FlowEstimate]]:
        if task is None:
            return {key: list(value) for key, value in self._history.items()}
        return {task: list(self._history.get(task, ())) }

    def metrics(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for task, signals in self._signals.items():
            if not signals:
                continue
            reward_total = sum(sig.reward for sig in signals)
            penalty_total = sum(sig.penalty for sig in signals)
            report[task] = {
                "samples": float(len(signals)),
                "avg_reward": reward_total / len(signals),
                "avg_penalty": penalty_total / len(signals),
                "net": (reward_total - penalty_total) / len(signals),
            }
        return report

    def state_dict(self) -> Dict[str, object]:
        return {
            "window": self.window,
            "smoothing": self.smoothing,
            "signals": {
                task: [signal.to_dict() for signal in signals]
                for task, signals in self._signals.items()
                if signals
            },
            "history": {
                task: [estimate.to_dict() for estimate in estimates]
                for task, estimates in self._history.items()
                if estimates
            },
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        window = payload.get("window")
        if isinstance(window, int) and window > 0:
            self.window = window
        smoothing = payload.get("smoothing")
        if isinstance(smoothing, (int, float)):
            self.smoothing = max(0.0, min(1.0, float(smoothing)))
        def _signals_factory() -> Deque[FlowSignal]:
            return deque(maxlen=self.window)
        self._signals = defaultdict(lambda: deque(maxlen=self.window))
        signals_payload = payload.get("signals", {})
        if isinstance(signals_payload, dict):
            for task, entries in signals_payload.items():
                dq = deque(maxlen=self.window)
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            dq.append(FlowSignal.from_dict(entry))
                self._signals[task] = dq
        self._history = defaultdict(deque)
        history_payload = payload.get("history", {})
        if isinstance(history_payload, dict):
            for task, entries in history_payload.items():
                dq = deque()
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            dq.append(FlowEstimate.from_dict(entry))
                self._history[task] = dq

    def _summarize(self, task: str, signals: Deque[FlowSignal]) -> FlowEstimate:
        support = len(signals)
        reward_sum = 0.0
        penalty_sum = 0.0
        novelty = 0.0
        retention = 0.0
        payoff = 0.0
        cost = 0.0
        novelty_bias = 0.0
        retention_bias = 0.0
        payoff_bias = 0.0
        cost_bias = 0.0

        decay = self._decay_factor(support)
        for idx, signal in enumerate(reversed(signals)):
            weight = self._weight(idx, decay)
            comps = signal.components
            reward_sum += weight * signal.reward
            penalty_sum += weight * signal.penalty
            novelty += weight * float(comps.get("novelty", 0.0))
            retention += weight * float(comps.get("retention", 0.0))
            payoff += weight * float(comps.get("payoff", 0.0))
            cost += weight * float(comps.get("cost", 0.0))
            novelty_bias += float(comps.get("novelty", 0.0))
            retention_bias += float(comps.get("retention", 0.0))
            payoff_bias += float(comps.get("payoff", 0.0))
            cost_bias += float(comps.get("cost", 0.0))

        norm = 1.0 / max(support, 1)
        gradients = {
            "novelty": novelty * norm,
            "retention": retention * norm,
            "payoff": payoff * norm,
            "cost": cost * norm,
        }
        bias = {
            "novelty": novelty_bias * norm,
            "retention": retention_bias * norm,
            "payoff": payoff_bias * norm,
            "cost": cost_bias * norm,
        }
        estimate = FlowEstimate(
            task=task,
            gradients=gradients,
            bias=bias,
            support=support,
            avg_reward=reward_sum * norm,
            avg_penalty=penalty_sum * norm,
        )
        return estimate

    def _decay_factor(self, support: int) -> float:
        if support <= 1:
            return 0.0
        return self.smoothing ** (1 / max(support - 1, 1))

    @staticmethod
    def _weight(idx: int, decay: float) -> float:
        return math.pow(decay, idx)
