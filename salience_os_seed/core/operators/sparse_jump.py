"""Shim selecting the appropriate SparseJump implementation."""

from __future__ import annotations

try:  # pragma: no cover - import guard
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

if getattr(torch, "__SALIENT_STUB__", False):  # pragma: no cover
    torch = None  # type: ignore

if torch is None:  # pragma: no cover - exercised in tests
    from ._sparse_jump_fallback import SparseJumpConfig, SparseJumpTeleporter
else:  # pragma: no cover - exercised when torch is installed
    from ._sparse_jump import SparseJumpConfig, SparseJumpTeleporter  # type: ignore

__all__ = ["SparseJumpConfig", "SparseJumpTeleporter"]
