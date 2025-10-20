"""SparseJump teleporter handling rare global attention hops.

The teleporter keeps a small per-sequence key-value cache that stores selected
hidden states. When the salience vector indicates AIM × NEW spikes, the
controller can request a teleport lookup. The module returns a residual that is
added to the current hidden state, approximating a selective global attention
fetch without paying the cost every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class SparseJumpConfig:
    """Configuration for the SparseJump teleporter."""

    cache_size: int = 32
    projection_dim: int = 256
    temperature: float = 0.25


@dataclass
class TeleportCache:
    """Per-sequence key-value cache."""

    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    def add(self, key: torch.Tensor, value: torch.Tensor, max_size: int) -> None:
        if len(self.keys) >= max_size:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(key.detach())
        self.values.append(value.detach())

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()


class SparseJumpTeleporter(nn.Module):
    """Implements teleport-style global hops triggered sparsely."""

    def __init__(self, config: SparseJumpConfig | None = None) -> None:
        super().__init__()
        self.config = config or SparseJumpConfig()
        self.key_proj = nn.Linear(self.config.projection_dim, self.config.projection_dim, bias=False)
        self.cache: Dict[int, TeleportCache] = {}

    def forward(
        self,
        token_states: torch.Tensor,
        sequence_id: int,
        trigger: bool,
    ) -> torch.Tensor:
        """Return residual from teleport cache lookups when triggered."""

        if not trigger:
            self._commit(sequence_id, token_states)
            return torch.zeros_like(token_states[:, -1])
        cache = self.cache.setdefault(sequence_id, TeleportCache())
        if not cache.keys:
            self._commit(sequence_id, token_states)
            return torch.zeros_like(token_states[:, -1])
        query = token_states[:, -1]  # shape (batch, dim)
        query_key = self.key_proj(query)
        keys = torch.stack(cache.keys, dim=0)  # (slots, batch, dim)
        values = torch.stack(cache.values, dim=0)
        scores = torch.einsum("sbd,bd->sb", keys, query_key) / self.config.temperature
        weights = torch.softmax(scores, dim=0)
        residual = torch.einsum("sb,sbd->bd", weights, values)
        self._commit(sequence_id, token_states)
        return residual

    def _commit(self, sequence_id: int, token_states: torch.Tensor) -> None:
        cache = self.cache.setdefault(sequence_id, TeleportCache())
        latest = token_states[:, -1]
        key = self.key_proj(latest)
        cache.add(key, latest, self.config.cache_size)

    def reset(self) -> None:
        self.cache.clear()
