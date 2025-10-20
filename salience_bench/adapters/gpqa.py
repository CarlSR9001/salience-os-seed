from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class GPQAAdapter(UnimplementedAdapter):
    """Stub adapter for the GPQA benchmark."""

    def __init__(self) -> None:
        super().__init__("GPQA")

    def prepare(self, workdir: Path, seed: int) -> None:
        super().prepare(workdir, seed)
        # TODO: Cache dataset locally (Hugging Face) and prepare choice labels.

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError("Implement multiple-choice evaluation for GPQA.")
