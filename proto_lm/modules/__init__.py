"""Extensible module registry for proto language model augmentations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Type

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModuleBlueprint:
    """Declarative specification for optional proto LM modules."""

    name: str
    position: str = "pre_core"
    config: Mapping[str, object] = field(default_factory=dict)


class SalienceModule(nn.Module):
    """Base class for modules influenced by salience signals."""

    def __init__(self, *, position: str = "pre_core") -> None:
        super().__init__()
        if position not in {"pre_core", "post_core"}:
            raise ValueError(f"Unsupported module position '{position}'")
        self._position = position

    @property
    def position(self) -> str:
        return self._position

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(tensor)

    def on_training_step(self, *, loss: float, snapshot: Mapping[str, object]) -> None:
        """Called after each optimiser step."""

    def on_salience_update(
        self,
        *,
        salience: Mapping[str, float],
        gating_decision: Optional[str],
        metrics: Mapping[str, object],
    ) -> None:
        """Called when fresh salience metrics are available."""

    def on_vocab_expand(self, *, new_size: int, delta: int) -> None:
        """Called when the base vocabulary expands."""


_REGISTRY: Dict[str, Type[SalienceModule]] = {}


def register_module(name: str) -> callable:
    """Decorator to register a salience-aware module class."""

    def _decorator(cls: Type[SalienceModule]) -> Type[SalienceModule]:
        if name in _REGISTRY:
            raise ValueError(f"Module '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def available_modules() -> Sequence[str]:
    return tuple(sorted(_REGISTRY))


class ModuleManager(nn.Module):
    """Runtime wrapper coordinating installed salience modules."""

    def __init__(
        self,
        *,
        embed_dim: int,
        blueprints: Optional[Sequence[ModuleBlueprint]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self._pre: nn.ModuleList = nn.ModuleList()
        self._post: nn.ModuleList = nn.ModuleList()
        for blueprint in blueprints or ():
            module = self._build_module(blueprint)
            if module.position == "pre_core":
                self._pre.append(module)
            else:
                self._post.append(module)

    def _build_module(self, blueprint: ModuleBlueprint) -> SalienceModule:
        cls = _REGISTRY.get(blueprint.name)
        if cls is None:
            raise KeyError(f"Unknown module '{blueprint.name}'")
        kwargs = dict(blueprint.config)
        kwargs.setdefault("position", blueprint.position)
        kwargs.setdefault("embed_dim", self.embed_dim)
        module = cls(**kwargs)
        if self.device is not None:
            module.to(self.device)
        return module

    def apply_pre(self, tensor: torch.Tensor) -> torch.Tensor:
        for module in self._pre:
            tensor = module.apply(tensor)
        return tensor

    def apply_post(self, tensor: torch.Tensor) -> torch.Tensor:
        for module in self._post:
            tensor = module.apply(tensor)
        return tensor

    def on_training_step(self, *, loss: float, snapshot: Mapping[str, object]) -> None:
        for module in self._iter_modules():
            module.on_training_step(loss=loss, snapshot=snapshot)

    def on_salience_update(
        self,
        *,
        salience: Mapping[str, float],
        gating_decision: Optional[str],
        metrics: Mapping[str, object],
    ) -> None:
        for module in self._iter_modules():
            module.on_salience_update(
                salience=salience,
                gating_decision=gating_decision,
                metrics=metrics,
            )

    def on_vocab_expand(self, *, new_size: int, delta: int) -> None:
        for module in self._iter_modules():
            module.on_vocab_expand(new_size=new_size, delta=delta)

    def _iter_modules(self) -> Iterable[SalienceModule]:
        yield from self._pre
        yield from self._post


# Eagerly import built-in modules so they register with the registry.
try:  # pragma: no cover - defensive against optional modules
    from . import self_awareness as _self_awareness  # noqa: F401
    from . import salience_compressor as _salience_compressor  # noqa: F401
except Exception:  # pragma: no cover - avoid hard import failure in minimal deployments
    _self_awareness = None
    _salience_compressor = None
