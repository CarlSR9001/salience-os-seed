"""Developer tooling for working with Salience OS Seed."""

from importlib import import_module
from typing import Any

__all__ = ["conversation_probe", "inspect_runtime", "microtraining_demo", "run_gym", "telemetry_replay"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
