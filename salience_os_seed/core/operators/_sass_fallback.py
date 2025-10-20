"""Pure Python fallback for the SASS operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass
class SASSConfig:
    d_model: int = 768
    state_channels: int = 1024
    kernel_size: int = 4
    num_layers: int = 4
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    use_checkpoint: bool = False


class SASSCore:
    """Fallback implementation that simply echoes inputs."""

    def __init__(self, config: SASSConfig) -> None:
        self.config = config
        self._hyper_appliers: list = []
        self.training = False

    def register_hyper_delta_applier(self, applier) -> None:  # pragma: no cover - compatibility hook
        self._hyper_appliers.append(applier)

    def train(self, mode: bool = True) -> "SASSCore":  # pragma: no cover - exercised in tests
        self.training = bool(mode)
        return self

    def eval(self) -> "SASSCore":  # pragma: no cover - exercised in tests
        return self.train(False)

    def __call__(
        self,
        hidden_states,
        layer_states: Optional[Sequence] = None,
        hyper_deltas: Optional[Iterable] = None,
        *,
        detach_states: bool = False,
    ):
        if hyper_deltas:
            for applier, delta in zip(self._hyper_appliers, hyper_deltas):
                try:
                    applier(delta)
                except Exception:
                    continue
        if layer_states is not None:
            states = [
                _detach_tensor(state) if detach_states or not self.training else state
                for state in layer_states
            ]
        else:
            states = []
            for _ in range(max(1, int(self.config.num_layers))):
                state = _clone_tensor(hidden_states)
                if detach_states or not self.training:
                    state = _detach_tensor(state)
                states.append(state)
        output = hidden_states
        return output, states


def _clone_tensor(tensor):
    if hasattr(tensor, "clone"):
        return tensor.clone()
    return tensor


def _detach_tensor(tensor):
    if hasattr(tensor, "detach"):
        detached = tensor.detach()
        if hasattr(detached, "requires_grad"):
            detached.requires_grad = False
        return detached
    if hasattr(tensor, "requires_grad"):
        tensor.requires_grad = False
    return tensor
