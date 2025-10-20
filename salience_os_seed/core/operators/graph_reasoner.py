"""Graph Reasoner head executing short message-passing bursts.

The head builds a temporary graph from structured memory tables and/or ad-hoc
entity extraction hints. It then runs a few iterations of message passing using
lightweight linear transforms, folding the aggregated graph context back into
the sequence hidden state as a low-rank residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn

from ..memory import StructuredMemory


@dataclass
class GraphReasonerConfig:
    """Configuration for the graph reasoner head."""

    node_dim: int = 384
    message_dim: int = 256
    iterations: int = 2
    residual_rank: int = 32


class GraphReasoner(nn.Module):
    """Short MPNN burst to enrich sequence representations."""

    def __init__(self, config: GraphReasonerConfig) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = nn.Linear(config.node_dim, config.message_dim)
        self.edge_encoder = nn.Linear(config.node_dim, config.message_dim)
        self.message_mlp = nn.Sequential(
            nn.LayerNorm(config.message_dim),
            nn.Linear(config.message_dim, config.message_dim),
            nn.GELU(),
            nn.Linear(config.message_dim, config.message_dim),
        )
        self.residual_projector = nn.Linear(config.message_dim, config.residual_rank)
        self.residual_expander = nn.Linear(config.residual_rank, config.node_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: StructuredMemory,
        entity_hints: Sequence[Mapping[str, object]] | None = None,
    ) -> Tuple[torch.Tensor, Mapping[str, object]]:
        graph = self._build_graph(memory, entity_hints)
        if not graph["nodes"]:
            return hidden_states, {"nodes": 0, "edges": 0}
        node_features = torch.stack(graph["nodes"], dim=0)
        edge_index = torch.stack(graph["edges"], dim=0) if graph["edges"] else None
        node_states = self.node_encoder(node_features)
        for _ in range(self.config.iterations):
            node_states = self._message_passing(node_states, edge_index)
        residual = self.residual_expander(self.residual_projector(node_states)).mean(dim=0, keepdim=True)
        hidden_states[:, -1, : self.config.node_dim] += residual
        return hidden_states, {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}

    def _build_graph(
        self,
        memory: StructuredMemory,
        entity_hints: Sequence[Mapping[str, object]] | None,
    ) -> Mapping[str, list[torch.Tensor]]:
        nodes: list[torch.Tensor] = []
        edges: list[torch.Tensor] = []
        for table in (memory.facts, memory.hypotheses, memory.todos):
            for record in table.iter():
                nodes.append(self._encode_text(record.text))
        if entity_hints:
            for hint in entity_hints:
                text = str(hint.get("text", ""))
                nodes.append(self._encode_text(text))
                parents = hint.get("links", [])
                for parent in parents:
                    edges.append(torch.tensor([len(nodes) - 1, parent], dtype=torch.long))
        return {"nodes": nodes, "edges": edges}

    def _message_passing(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return node_states
        src, dst = edge_index[:, 0], edge_index[:, 1]
        messages = self.message_mlp(node_states[src])
        agg = torch.zeros_like(node_states)
        agg.index_add_(0, dst, messages)
        updated = torch.tanh(node_states + agg)
        return updated

    def _encode_text(self, text: str) -> torch.Tensor:
        # Cheap hashing-based embedding for the seed implementation; replace with
        # learned encoder during training.
        rng = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
        return torch.randn(self.config.node_dim, generator=rng)
