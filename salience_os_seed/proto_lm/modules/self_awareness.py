"""Self-awareness module adapting representations using salience telemetry."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn

from . import SalienceModule, register_module


_TRACKED_KEYS: tuple[str, ...] = (
    "uncertainty",
    "novelty",
    "progress",
    "roi",
    "drag",
    "cost",
    "alignment",
)


@register_module("self_awareness")
class SelfAwarenessModule(SalienceModule):
    """Adjust hidden representations based on salience and training telemetry."""

    def __init__(
        self,
        *,
        embed_dim: int,
        smoothing: float = 0.1,
        learning_rate: float = 0.01,
        position: str = "post_core",
    ) -> None:
        super().__init__(position=position)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.context_gate = nn.Parameter(torch.zeros(embed_dim))
        self.smoothing = float(max(1e-4, min(0.5, smoothing)))
        self.learning_rate = float(max(1e-4, learning_rate))
        self.register_buffer("salience_state", torch.zeros(len(_TRACKED_KEYS)))
        self.register_buffer("loss_ema", torch.zeros(1))
        self._last_gate: Optional[str] = None

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        normalized = self.layer_norm(tensor)
        gate = torch.tanh(self.context_gate).view(1, 1, -1)
        return normalized + gate

    def on_training_step(self, *, loss: float, snapshot: Mapping[str, object]) -> None:
        with torch.no_grad():
            loss_tensor = torch.tensor(float(loss), device=self.context_gate.device)
            self.loss_ema.mul_(1.0 - self.smoothing).add_(loss_tensor * self.smoothing)
            residual = torch.tanh(loss_tensor - self.loss_ema)
            self.context_gate.data.add_(self.learning_rate * residual)
            self.context_gate.data.clamp_(-3.0, 3.0)

    def on_salience_update(
        self,
        *,
        salience: Mapping[str, float],
        gating_decision: Optional[str],
        metrics: Mapping[str, object],
    ) -> None:
        del metrics  # Unused for now but reserved for future heuristics.
        with torch.no_grad():
            for idx, key in enumerate(_TRACKED_KEYS):
                value = torch.tensor(float(salience.get(key, 0.0)), device=self.salience_state.device)
                self.salience_state[idx] = (
                    (1.0 - self.smoothing) * self.salience_state[idx] + self.smoothing * value
                )
            guide = torch.zeros_like(self.context_gate)
            limit = min(len(_TRACKED_KEYS), guide.numel())
            guide[:limit] = self.salience_state[:limit].to(self.context_gate.device)
            if gating_decision == "DROP":
                guide.neg_()
            self.context_gate.data.add_(self.learning_rate * guide)
            self.context_gate.data.clamp_(-3.0, 3.0)
            self._last_gate = gating_decision

    def on_vocab_expand(self, *, new_size: int, delta: int) -> None:
        del new_size, delta
        if self.context_gate.numel() != self.layer_norm.normalized_shape[0]:
            return
        with torch.no_grad():
            self.context_gate.data.mul_(0.95)
