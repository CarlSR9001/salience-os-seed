"""Top-level Salience OS Seed package.

This module centralizes the public namespace so that the repository can be
installed or imported as a conventional Python package.  Key functional areas
(such as :mod:`core`, :mod:`runtime`, and :mod:`conversation`) are re-exported
for convenience while keeping the implementation modules organised below the
``salience_os_seed`` package directory.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "adaptive",
    "conversation",
    "core",
    "ingestion",
    "proto_lm",
    "rlm",
    "runtime",
    "telemetry",
    "tools",
    "training",
]


def _load_submodule(name: str) -> ModuleType:
    """Lazy loader used by ``__getattr__`` and ``__dir__``."""

    return import_module(f"{__name__}.{name}")


if TYPE_CHECKING:  # pragma: no cover - import side-effects only for typing
    from . import adaptive, conversation, core, ingestion, proto_lm, rlm, runtime, tools, training
    from . import telemetry as telemetry


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = _load_submodule(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


telemetry = _load_submodule("telemetry")
