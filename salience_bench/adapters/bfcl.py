from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class BFCLAdapter(UnimplementedAdapter):
    """Stub adapter for Berkeley Function-Calling Leaderboard."""

    def __init__(self) -> None:
        super().__init__("BFCL")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Download BFCL tasks and tool specifications.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Implement BFCL function-calling evaluation.")
