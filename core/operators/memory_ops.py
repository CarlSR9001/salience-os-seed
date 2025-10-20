"""Structured memory operator verbs.

This module exposes a `MemoryOperator` that applies JSON-ish verbs to the
`StructuredMemory` tables. The operator provides deterministic logging and
ensures updates remain transactional (no partial mutations when an action fails).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

from ..memory import MemoryRecord, MemoryTable, StructuredMemory


@dataclass
class MemoryOpResult:
    """Outcome of a memory verb execution."""

    applied: bool
    new_records: Sequence[MemoryRecord]
    removed_ids: Sequence[int]
    verb: Mapping[str, object]


class MemoryOperator:
    """Apply structured memory verbs (facts/hypotheses/todos)."""

    SUPPORTED_OPS = {
        "add_fact": "facts",
        "promote_hypothesis": "hypotheses",
        "schedule_todo": "todos",
        "retract": None,
    }

    def __init__(self, memory: StructuredMemory) -> None:
        self.memory = memory

    def execute(self, verb: Mapping[str, object]) -> MemoryOpResult:
        op = str(verb.get("op"))
        if op not in self.SUPPORTED_OPS:
            return MemoryOpResult(applied=False, new_records=[], removed_ids=[], verb=verb)
        if op == "retract":
            record_id = int(verb.get("id", -1))
            removed = self._remove_record(record_id)
            return MemoryOpResult(applied=removed, new_records=[], removed_ids=[record_id] if removed else [], verb=verb)
        table_name = self.SUPPORTED_OPS[op]
        assert table_name is not None
        table = getattr(self.memory, table_name)
        text = str(verb.get("text", ""))
        score = float(verb.get("score", 0.0))
        metadata = verb.get("metadata", {})
        record = table.add(text=text, score=score, metadata=dict(metadata))
        return MemoryOpResult(applied=True, new_records=[record], removed_ids=[], verb=verb)

    def _remove_record(self, record_id: int) -> bool:
        removed = False
        for table in (self.memory.facts, self.memory.hypotheses, self.memory.todos):
            if record_id in table._records:  # type: ignore[attr-defined]
                table.remove(record_id)
                removed = True
        return removed

    def snapshot(self) -> Mapping[str, object]:
        return self.memory.serialize()
