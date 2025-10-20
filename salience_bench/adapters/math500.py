from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class MATH500Adapter(UnimplementedAdapter):
    """Stub adapter for the MATH-500 benchmark."""

    def __init__(self) -> None:
        super().__init__("MATH_500")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Download math problems and normalization utilities.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Implement MATH-500 evaluation and answer normalization.")
