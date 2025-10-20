"""Maintenance management for runtime memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from ..core.memory import (
    MaintenanceThresholds,
    StructuredMemory,
    archive_low_roi_facts,
    merge_redundant_entries,
    prune_failed_hypotheses,
    should_cleanup,
    summarize_old_context,
)
from ..core.memory.maintenance import ArchiveStore


@dataclass(slots=True)
class MaintenanceManager:
    """Runs cleanup routines when thresholds are triggered."""

    memory: StructuredMemory
    thresholds: MaintenanceThresholds
    store: ArchiveStore

    def run(self, salience: Mapping[str, float]) -> Dict[str, int]:
        if not should_cleanup(salience, self.memory, self.thresholds):
            return {}
        report: Dict[str, int] = {}
        archived = archive_low_roi_facts(self.memory, threshold=self.thresholds.low_roi_threshold, store=self.store)
        if archived:
            report["archived"] = archived
        merged = merge_redundant_entries(self.memory, self.thresholds.merge_similarity)
        if merged:
            report["merged"] = merged
        pruned = prune_failed_hypotheses(self.memory, self.thresholds.verification_failure_limit)
        if pruned:
            report["pruned"] = pruned
        summarized = summarize_old_context(
            self.memory,
            age_threshold=self.thresholds.summary_age_threshold,
            chunk_size=self.thresholds.summary_chunk,
        )
        if summarized:
            report["summarized"] = summarized
        return report


def default_archive_store(path: Optional[str] = None) -> ArchiveStore:
    root = Path(path).expanduser() if path else Path("storage/memory_archive")
    return ArchiveStore(root=root)


__all__ = ["MaintenanceManager", "default_archive_store"]
