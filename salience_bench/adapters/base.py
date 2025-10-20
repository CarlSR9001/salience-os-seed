from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ..core import AggregateReport, BenchmarkAdapter, MetricSummary, ModelConfig, SimpleAggregateAdapter

logger = logging.getLogger(__name__)


class UnimplementedAdapter(SimpleAggregateAdapter):
    """Adapter skeleton that marks TODO implementations."""

    def __init__(self, name: str) -> None:
        self.name = name

    def prepare(self, workdir: Path, seed: int) -> None:  # noqa: D401 - simple implementation
        logger.warning("prepare() for %s not yet implemented", self.name)

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError(f"Benchmark adapter '{self.name}' run() not implemented yet")


def aggregate_placeholder(name: str, runs: Sequence[MetricSummary]) -> AggregateReport:
    adapter = UnimplementedAdapter(name)
    return adapter.aggregate(runs)
