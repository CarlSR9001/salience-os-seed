"""Salience-driven compressor module for proto language model embeddings."""

from __future__ import annotations

from typing import Mapping, Optional

import math

import torch
import torch.nn as nn

from . import SalienceModule, register_module


@register_module("salience_compressor")
class SalienceCompressorModule(SalienceModule):
    """Applies a learned low-rank bottleneck modulated by salience strength."""

    def __init__(
        self,
        *,
        embed_dim: int,
        bottleneck_ratio: float = 0.5,
        strength_lr: float = 0.01,
        smoothing: float = 0.05,
        position: str = "pre_core",
    ) -> None:
        super().__init__(position=position)
        bottleneck = max(1, int(embed_dim * max(0.05, min(1.0, bottleneck_ratio))))
        self.down = nn.Linear(embed_dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        self.strength = nn.Parameter(torch.zeros(1))
        self.strength_lr = float(max(1e-4, strength_lr))
        self.smoothing = float(max(1e-4, min(0.2, smoothing)))
        self.register_buffer("energy_ema", torch.zeros(1))
        self.register_buffer("grad_ema", torch.zeros(1))

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        compressed = self.down(tensor)
        reconstructed = self.up(compressed)
        strength = torch.tanh(self.strength).view(1, 1, 1)
        return tensor + strength * reconstructed

    def on_training_step(self, *, loss: float, snapshot: Mapping[str, object]) -> None:
        grad_health = snapshot.get("grad_health", {}) if isinstance(snapshot, Mapping) else {}
        grad_norm = float(grad_health.get("grad_norm", 0.0))
        with torch.no_grad():
            grad_tensor = torch.tensor(grad_norm, device=self.strength.device)
            self.grad_ema.mul_(1.0 - self.smoothing).add_(self.smoothing * grad_tensor)
            delta = grad_tensor - self.grad_ema
            self.strength.data.add_(self.strength_lr * delta.clamp(-1.0, 1.0))
            self.strength.data.clamp_(-2.0, 2.0)

    def on_salience_update(
        self,
        *,
        salience: Mapping[str, float],
        gating_decision: Optional[str],
        metrics: Mapping[str, object],
    ) -> None:
        del metrics
        energy = 0.0
        for value in salience.values():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            energy += val * val
        with torch.no_grad():
            energy_tensor = torch.tensor(energy, device=self.strength.device)
            self.energy_ema.mul_(1.0 - self.smoothing).add_(self.smoothing * energy_tensor)
            delta = energy_tensor - self.energy_ema
            if gating_decision == "DROP":
                delta = -delta
            self.strength.data.add_(self.strength_lr * delta.clamp(-1.0, 1.0))
            self.strength.data.clamp_(-2.5, 2.5)

    def on_vocab_expand(self, *, new_size: int, delta: int) -> None:
        del new_size, delta
        # Soften influence after capacity growth to avoid overfitting to stale residuals.
        with torch.no_grad():
            self.strength.data.mul_(0.98)
