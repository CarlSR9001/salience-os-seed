"""Sensor pipeline wrapper for the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

if getattr(torch, "__SALIENT_STUB__", False):  # pragma: no cover
    torch = None  # type: ignore

from ..core.sensors import SensorBank
from ..core.reflection import Scratchpad


@dataclass(slots=True)
class SensorPipeline:
    """Encapsulates sensor state enrichment and execution."""

    sensor_bank: SensorBank = field(default_factory=SensorBank.default_bank)
    _sensor_memory: MutableMapping[str, object] = field(default_factory=dict)

    @property
    def salience_dim(self) -> int:
        return len(self.sensor_bank.ordering)

    def update_memory_snapshot(self, snapshot: Mapping[str, object]) -> None:
        for key, value in snapshot.items():
            self._sensor_memory[key] = value

    def run(
        self,
        state: Mapping[str, object],
        meta_snapshot: Mapping[str, float],
    ) -> tuple[Mapping[str, float], object]:
        vector = self.sensor_bank.tick(state, self._sensor_memory, meta_snapshot)
        return vector.as_mapping(), vector

    def enrich_state(
        self,
        state: Mapping[str, object],
        *,
        scratchpad: Scratchpad,
        hidden_states,
        controller_last_action,
    ) -> Dict[str, object]:
        enriched: Dict[str, object] = dict(state)
        if controller_last_action is not None:
            enriched.setdefault("last_action", controller_last_action.operator.name)
            enriched.setdefault("last_action_depth", controller_last_action.cot_depth)
        if "scratchpad_text" not in enriched:
            if scratchpad.current_trace:
                enriched["scratchpad_text"] = " | ".join(scratchpad.current_trace)
            else:
                enriched["scratchpad_text"] = scratchpad.summarize(max_traces=1)
        if "scratchpad" not in enriched:
            enriched["scratchpad"] = list(scratchpad.current_trace)
        if "contradictions" not in enriched:
            enriched["contradictions"] = 0.0
        if hidden_states is not None and "embedding" not in enriched:
            enriched["embedding"] = self._hidden_embedding_vector(hidden_states)
        return enriched

    @staticmethod
    def _hidden_embedding_vector(tensor) -> object:
        if torch is None:
            return []
        try:
            return tensor.detach().cpu().numpy().flatten()[-256:]
        except Exception:  # pragma: no cover - defensive
            return []
