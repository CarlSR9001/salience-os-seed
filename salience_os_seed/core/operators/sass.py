"""Shim selecting the appropriate SASS implementation."""

from __future__ import annotations

try:  # pragma: no cover - import guard
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

if getattr(torch, "__SALIENT_STUB__", False):  # pragma: no cover
    torch = None  # type: ignore

if torch is None:  # pragma: no cover - exercised in tests
    from ._sass_fallback import SASSConfig, SASSCore
else:  # pragma: no cover - exercised when torch is installed
    from ._sass_torch import SASSConfig, SASSCore

__all__ = ["SASSConfig", "SASSCore"]
