"""Fallback SparseJump teleporter that avoids torch dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass
class SparseJumpConfig:
    cache_size: int = 32
    projection_dim: int = 256
    temperature: float = 0.25


@dataclass
class TeleportCache:
    keys: List[Mapping[str, float]] = field(default_factory=list)
    values: List[Mapping[str, float]] = field(default_factory=list)

    def add(self, key: Mapping[str, float], value: Mapping[str, float], max_size: int) -> None:
        if len(self.keys) >= max_size:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(dict(key))
        self.values.append(dict(value))

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()


class SparseJumpTeleporter:
    def __init__(self, config: SparseJumpConfig | None = None) -> None:
        self.config = config or SparseJumpConfig()
        self.cache: Dict[int, TeleportCache] = {}

    def __call__(
        self,
        token_states,
        *,
        sequence_id: int = 0,
        trigger: bool = False,
    ):
        cache = self.cache.setdefault(sequence_id, TeleportCache())
        if not trigger or not cache.keys:
            self._commit(cache, token_states)
            return {}
        latest = self._extract_vector(token_states)
        result = dict(latest)
        self._commit(cache, token_states)
        return result

    def _commit(self, cache: TeleportCache, token_states) -> None:
        vector = self._extract_vector(token_states)
        cache.add(vector, vector, self.config.cache_size)

    @staticmethod
    def _extract_vector(token_states) -> Mapping[str, float]:
        if isinstance(token_states, Mapping):
            return token_states
        return {"value": float(len(token_states) if token_states else 0.0)}

    def reset(self) -> None:
        self.cache.clear()
