"""Structured external memory package for SalienceOS Seed."""

from .maintenance import (
    MaintenanceThresholds,
    archive_low_roi_facts,
    merge_redundant_entries,
    prune_failed_hypotheses,
    should_cleanup,
    summarize_old_context,
)
from .tables import MemoryRecord, MemoryTable, StructuredMemory

__all__ = [
    "MemoryRecord",
    "MemoryTable",
    "StructuredMemory",
    "MaintenanceThresholds",
    "archive_low_roi_facts",
    "merge_redundant_entries",
    "prune_failed_hypotheses",
    "should_cleanup",
    "summarize_old_context",
]
