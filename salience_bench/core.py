from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration describing the model under evaluation."""

    name: str
    revision: str | None = None
    checkpoint_path: Path | None = None
    generation_kwargs: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True)
class MetricSummary:
    """Container for benchmark metric outputs."""

    benchmark: str
    seed: int
    metrics: Mapping[str, float]
    metadata: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True)
class AggregateReport:
    """Aggregate results across multiple seeds."""

    benchmark: str
    mean_metrics: Mapping[str, float]
    std_metrics: Mapping[str, float]
    ci95_metrics: Mapping[str, tuple[float, float]]
    runs: Sequence[MetricSummary]


class BenchmarkAdapter(Protocol):
    """Interface for all benchmark adapters."""

    name: str

    def prepare(self, workdir: Path, seed: int) -> None:
        """Prepare any benchmark-specific assets before execution."""

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        """Execute the benchmark for a single seed and return metrics."""

    def aggregate(self, runs: Sequence[MetricSummary]) -> AggregateReport:
        """Aggregate multi-seed results into a report."""


class SimpleAggregateAdapter(BenchmarkAdapter, abc.ABC):
    """Helper base class for adapters using arithmetic mean aggregation."""

    name: str

    def aggregate(self, runs: Sequence[MetricSummary]) -> AggregateReport:
        if not runs:
            raise ValueError("No runs provided for aggregation")
        metrics = runs[0].metrics.keys()
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        cis: dict[str, tuple[float, float]] = {}
        for metric in metrics:
            series = [run.metrics[metric] for run in runs]
            mean = sum(series) / len(series)
            variance = sum((value - mean) ** 2 for value in series) / max(len(series) - 1, 1)
            std = variance ** 0.5
            ci_low = mean - 1.96 * std / (len(series) ** 0.5)
            ci_high = mean + 1.96 * std / (len(series) ** 0.5)
            means[metric] = mean
            stds[metric] = std
            cis[metric] = (ci_low, ci_high)
        return AggregateReport(
            benchmark=self.name,
            mean_metrics=means,
            std_metrics=stds,
            ci95_metrics=cis,
            runs=runs,
        )
