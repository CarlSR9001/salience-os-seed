"""Memory maintenance utilities for ``StructuredMemory``.

The helpers defined here keep structured memory bounded so salience drag does
not grow unchecked. They implement the behaviours outlined in
``docs/tier2_design_brief.md`` and are intentionally lightweight so they can run
inside the main runtime loop without introducing heavy dependencies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableSequence, Sequence

from .tables import MemoryRecord, StructuredMemory

ARCHIVE_ROOT = Path("storage/memory_archive")
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class MaintenanceThresholds:
    """Tunables for cleanup triggers and limits."""

    drag_trigger: float = 0.5
    facts_max: int = 512
    hypotheses_max: int = 256
    todos_max: int = 128
    verification_failure_limit: int = 3
    merge_similarity: float = 0.75
    summary_chunk: int = 10
    summary_age_threshold: int = 1000
    low_roi_threshold: float = 0.1


def should_cleanup(
    salience: Mapping[str, float],
    memory: StructuredMemory,
    thresholds: MaintenanceThresholds | None = None,
) -> bool:
    """Return ``True`` when cleanup routines should run."""

    t = thresholds or MaintenanceThresholds()
    drag = float(salience.get("drag", 0.0))
    if drag >= t.drag_trigger:
        return True
    if memory.facts.count() > t.facts_max:
        return True
    if memory.hypotheses.count() > t.hypotheses_max:
        return True
    if memory.todos.count() > t.todos_max:
        return True
    for record in memory.hypotheses.iter():
        if _metadata_int(record.metadata, "failures") >= t.verification_failure_limit:
            return True
    return False


def archive_low_roi_facts(memory: StructuredMemory, threshold: float = 0.1) -> int:
    """Move low-ROI facts into cold storage and remove them from hot tables."""

    archived = 0
    to_remove: MutableSequence[int] = []
    timestamp = int(time.time())
    archive_path = ARCHIVE_ROOT / f"facts_{timestamp}.jsonl"
    with archive_path.open("a", encoding="utf-8") as handle:
        for record in list(memory.facts.iter()):
            score = float(record.metadata.get("roi", record.score))
            if score < threshold:
                handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                to_remove.append(record.id)
                archived += 1
    for record_id in to_remove:
        memory.facts.remove(record_id)
    return archived


def merge_redundant_entries(memory: StructuredMemory, similarity_threshold: float = 0.75) -> int:
    """Merge similar facts or hypotheses by averaging their metadata."""

    merged = 0
    merged += _merge_table(memory.facts, similarity_threshold)
    merged += _merge_table(memory.hypotheses, similarity_threshold)
    return merged


def prune_failed_hypotheses(memory: StructuredMemory, failure_limit: int = 3) -> int:
    """Remove hypotheses that repeatedly failed verification."""

    removed = 0
    for record in list(memory.hypotheses.iter()):
        if _metadata_int(record.metadata, "failures") >= failure_limit:
            memory.hypotheses.remove(record.id)
            removed += 1
    return removed


def summarize_old_context(memory: StructuredMemory, age_threshold: int = 1000, chunk_size: int = 10) -> int:
    """Compress older todos and facts into summary entries."""

    summary_count = 0
    summary_count += _summarize_table(memory.facts, age_threshold, "facts", chunk_size=chunk_size)
    summary_count += _summarize_table(memory.todos, age_threshold, "todos", chunk_size=chunk_size)
    return summary_count


def _metadata_int(metadata: Mapping[str, float], key: str) -> int:
    value = metadata.get(key, 0.0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _merge_table(table, similarity_threshold: float) -> int:
    records = list(table.iter())
    merged = 0
    seen: set[int] = set()
    for i, record in enumerate(records):
        if record.id in seen:
            continue
        cluster = [record]
        for other in records[i + 1 :]:
            if other.id in seen:
                continue
            similarity = _text_similarity(record.text, other.text)
            if similarity >= similarity_threshold:
                cluster.append(other)
                seen.add(other.id)
        if len(cluster) > 1:
            merged_record = _merge_records(cluster)
            table.update(
                record.id,
                text=merged_record.text,
                score=merged_record.score,
                metadata=merged_record.metadata,
            )
            for extra in cluster[1:]:
                table.remove(extra.id)
            merged += len(cluster) - 1
    return merged


def _merge_records(records: Sequence[MemoryRecord]) -> MemoryRecord:
    if not records:
        raise ValueError("records must be non-empty")
    text = _summarize_text(r.text for r in records)
    score = float(sum(r.score for r in records) / len(records))
    metadata = {}
    keys = {key for record in records for key in record.metadata.keys()}
    for key in keys:
        values = [float(record.metadata.get(key, 0.0)) for record in records]
        if key == "failures":
            metadata[key] = max(values)
        else:
            metadata[key] = sum(values) / len(values)
    return MemoryRecord(id=records[0].id, text=text, score=score, metadata=metadata)


def _summarize_table(table, age_threshold: int, prefix: str, *, chunk_size: int) -> int:
    cutoff = getattr(table, "_next_id", 0) - age_threshold
    if cutoff <= 0:
        return 0
    old_records = [record for record in table.iter() if record.id <= cutoff]
    if not old_records:
        return 0
    count = 0
    for index in range(0, len(old_records), max(1, chunk_size)):
        chunk = old_records[index : index + chunk_size]
        summary_text = _summarize_text(record.text for record in chunk)
        table.add(f"{prefix}_summary::{summary_text[:256]}", score=float(len(chunk)))
        for record in chunk:
            table.remove(record.id)
        count += len(chunk)
    return count


def _summarize_text(texts: Iterable[str]) -> str:
    return " ".join(text.strip() for text in texts if text.strip())


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union
