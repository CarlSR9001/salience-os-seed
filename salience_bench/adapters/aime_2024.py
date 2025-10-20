from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class AIME2024Adapter(UnimplementedAdapter):
    """Stub adapter for the AIME 2024 benchmark."""

    def __init__(self) -> None:
        super().__init__("AIME_2024")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Download dataset split and cache locally if needed.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Implement short-answer evaluation for AIME 2024.")
