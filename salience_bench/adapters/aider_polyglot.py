from __future__ import annotations

from pathlib import Path

from ..core import MetricSummary, ModelConfig
from .base import UnimplementedAdapter


class AiderPolyglotAdapter(UnimplementedAdapter):
    """Placeholder adapter for the Aider Polyglot benchmark."""

    def __init__(self) -> None:
        super().__init__("Aider_Polyglot")

    def prepare(self, workdir: Path, seed: int) -> None:  # pragma: no cover - stub
        super().prepare(workdir, seed)

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:  # pragma: no cover - stub
        return super().run(model, seed, output_dir)
