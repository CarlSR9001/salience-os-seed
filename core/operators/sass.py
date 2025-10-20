"""Salience-Addressable State Space (SASS) core implementation.

The design takes inspiration from recent state-space models (e.g., Mamba). The
implementation keeps the math approachable while preserving the core benefits:
linear-time token processing with strong locality. Teleport KV integration is
handled externally via `SparseJumpTeleporter` which writes to/reads from a tiny
key-value cache when salience signals warrant a global hop.

Key characteristics:
- Stack of `StateSpaceBlock`s (lightweight GRU-style gating + convolutional skip).
- RoPE positional embedding injection for stability on long contexts.
- Optional gradient checkpoint hook via `torch.utils.checkpoint` in training.
- Hooks for low-rank hyper-adapter deltas (rank ≤ 8) loaded per step to avoid
  inflating the base model footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASSConfig:
    """Configuration for the SASS core."""

    d_model: int = 768
    state_channels: int = 1024
    kernel_size: int = 4
    num_layers: int = 20
    dropout: float = 0.05
    rope_theta: float = 10_000.0
    use_checkpoint: bool = False


class StateSpaceBlock(nn.Module):
    """Single state-space processing block with gated updates."""

    def __init__(self, config: SASSConfig) -> None:
        super().__init__()
        d_model = config.d_model
        d_state = config.state_channels
        k = config.kernel_size
        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, 3 * d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        self.conv = nn.Conv1d(d_state, d_state, kernel_size=k, padding=k - 1, groups=d_state)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("rope_cache", None, persistent=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, d_model)
        residual = x
        x = self.norm(x)
        gates, update, mix = self.input_proj(x).chunk(3, dim=-1)
        gates = torch.sigmoid(gates)
        mix = torch.tanh(mix)
        # Transpose for convolution: (batch, d_state, seq)
        conv_in = (update * mix).transpose(1, 2)
        conv_out = self.conv(conv_in)
        conv_out = conv_out.transpose(1, 2)
        if conv_out.size(1) != gates.size(1):
            conv_out = conv_out[:, : gates.size(1), :]
        if state is None:
            state = torch.zeros_like(conv_out[:, 0:1, :])
        # Recurrent-style blending
        new_state = gates * conv_out + (1.0 - gates) * state
        out = self.output_proj(new_state)
        out = self.dropout(out)
        x = residual + out
        return x, new_state.detach()


class SASSCore(nn.Module):
    """Stacked state-space backbone with optional hyper-adapter deltas."""

    def __init__(self, config: SASSConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([StateSpaceBlock(config) for _ in range(config.num_layers)])
        self._rope_cache: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hyper_delta_appliers: list[Callable[[nn.Module], None]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_states: Optional[Sequence[torch.Tensor]] = None,
        hyper_deltas: Optional[Iterable[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        seq_len = hidden_states.size(1)
        cos, sin = self._get_rope_factors(seq_len, hidden_states.device, hidden_states.dtype)
        hidden_states = self._apply_rope(hidden_states, cos, sin)

        if hyper_deltas is not None:
            self._apply_hyper_deltas(hyper_deltas)

        new_states: list[torch.Tensor] = []
        states_iter = list(layer_states) if layer_states is not None else [None] * len(self.layers)
        for idx, (layer, prev_state) in enumerate(zip(self.layers, states_iter)):
            if self.config.use_checkpoint and self.training:
                hidden_states, layer_state = torch.utils.checkpoint.checkpoint(layer, hidden_states, prev_state)
            else:
                hidden_states, layer_state = layer(hidden_states, prev_state)
            new_states.append(layer_state)
        return hidden_states, new_states

    def register_hyper_delta_applier(self, applier: Callable[[nn.Module], None]) -> None:
        """Register a callback that injects low-rank deltas into the core."""

        self.hyper_delta_appliers.append(applier)

    def _apply_hyper_deltas(self, deltas: Iterable[torch.Tensor]) -> None:
        for applier, delta in zip(self.hyper_delta_appliers, deltas):
            applier(delta)

    def _apply_rope(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if hidden_states.size(-1) % 2 != 0:
            raise ValueError("SASSCore requires even hidden dimension for ROPE application")
        q, k = hidden_states.chunk(2, dim=-1)
        cos = cos.to(hidden_states.device, hidden_states.dtype)
        sin = sin.to(hidden_states.device, hidden_states.dtype)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return torch.cat([q_rot, k_rot], dim=-1)

    def _get_rope_factors(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, device, dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]
        half_dim = self.config.d_model // 2
        if half_dim % 2 != 0:
            raise ValueError("SASSCore requires hidden dimension divisible by 4 for ROPE")
        position = torch.arange(seq_len, device=device, dtype=dtype)
        inv_freq = 1.0 / (
            self.config.rope_theta ** (
                torch.arange(0, half_dim, 2, device=device, dtype=dtype) / half_dim
            )
        )
        angles = torch.einsum("i,j->ij", position, inv_freq)
        cos = torch.zeros(seq_len, half_dim, device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        cos[:, 0::2] = torch.cos(angles)
        cos[:, 1::2] = torch.cos(angles)
        sin[:, 0::2] = torch.sin(angles)
        sin[:, 1::2] = torch.sin(angles)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
