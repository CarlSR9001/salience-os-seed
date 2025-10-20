"""Benchmark adapter registry."""

from __future__ import annotations

from .aider_polyglot import AiderPolyglotAdapter
from .aime_2024 import AIME2024Adapter
from .bfcl import BFCLAdapter
from .grind import GRINDAdapter
from .gpqa import GPQAAdapter
from .math500 import MATH500Adapter
from .stub import StubAdapter
from .swe_bench import SWEBenchAdapter

__all__ = [
    "AiderPolyglotAdapter",
    "AIME2024Adapter",
    "BFCLAdapter",
    "GRINDAdapter",
    "GPQAAdapter",
    "MATH500Adapter",
    "StubAdapter",
    "SWEBenchAdapter",
]

REGISTRY = {
    "Aider_Polyglot": AiderPolyglotAdapter,
    "AIME_2024": AIME2024Adapter,
    "BFCL": BFCLAdapter,
    "GRIND": GRINDAdapter,
    "GPQA": GPQAAdapter,
    "MATH_500": MATH500Adapter,
    "stub": StubAdapter,
    "SWE_bench": SWEBenchAdapter,
}
