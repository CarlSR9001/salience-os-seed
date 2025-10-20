"""Fallback graph reasoner that operates without torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence, Tuple

from ..memory import StructuredMemory


@dataclass
class GraphReasonerConfig:
    node_dim: int = 384
    message_dim: int = 256
    iterations: int = 2
    residual_rank: int = 32


class GraphReasoner:
    """Lightweight placeholder that reports simple graph statistics."""

    def __init__(self, config: GraphReasonerConfig | None = None) -> None:
        self.config = config or GraphReasonerConfig()

    def __call__(
        self,
        hidden_states,
        memory: StructuredMemory,
        entity_hints: Sequence[Mapping[str, object]] | None = None,
    ) -> Tuple[object, Mapping[str, int]]:
        node_count = 0
        for table in (memory.facts, memory.hypotheses, memory.todos):
            node_count += sum(1 for _ in table.iter())
        if entity_hints:
            node_count += len(entity_hints)
        stats: MutableMapping[str, int] = {"nodes": node_count, "edges": 0}
        return hidden_states, stats
