from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class SWEBenchAdapter(UnimplementedAdapter):
    """Stub adapter for SWE-bench / SWE-bench Verified."""

    def __init__(self) -> None:
        super().__init__("SWE_bench")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Set up SWE-bench harness (Lite or Verified). Ensure repos cached per task.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Integrate SWE-bench execution and scoring pipeline.")
