from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class GRINDAdapter(UnimplementedAdapter):
    """Placeholder for Vellum GRIND harness integration."""

    def __init__(self) -> None:
        super().__init__("GRIND")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Download prompt templates and configure Vellum credentials.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Integrate Vellum GRIND API client and scoring.")
