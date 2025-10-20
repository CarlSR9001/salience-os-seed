from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class StubAdapter(UnimplementedAdapter):
    """Minimal adapter for CLI smoke testing."""

    def __init__(self) -> None:
        super().__init__("stub")

    def prepare(self, workdir: Path, seed: int) -> None:
        workdir.mkdir(parents=True, exist_ok=True)

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        return MetricSummary(benchmark=self.name, seed=seed, metrics={"score": 0.0})
