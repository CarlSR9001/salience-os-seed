"""Shim selecting the appropriate proto language model implementation."""

from __future__ import annotations

try:  # pragma: no cover - import guard
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

if getattr(torch, "__SALIENT_STUB__", False):  # pragma: no cover - fallback to stub
    torch = None  # type: ignore

if torch is None:  # pragma: no cover - exercised indirectly via tests
    from ._fallback import CheckpointRecord, ProtoLanguageModel, TrainingConfig
else:  # pragma: no cover - exercised when torch is installed
    from ._torch_impl import ProtoLanguageModel
    from ._fallback import TrainingConfig
    from .checkpoints import CheckpointRecord

__all__ = ["TrainingConfig", "ProtoLanguageModel", "CheckpointRecord"]
