"""Structured external memory tables for SalienceOS Seed.

The architecture relies on explicit tables for `facts`, `hypotheses`, and
`todos`. Each table offers transactional edits producing immutable snapshots so
operators can diff state efficiently. The implementation is intentionally kept
in pure Python with standard library data structures to stay lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class MemoryRecord:
    """Immutable record stored in a memory table."""

    id: int
    text: str
    score: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class MemoryTable:
    """Append-only table with transaction-friendly operations."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._records: Dict[int, MemoryRecord] = {}
        self._next_id = 0

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def iter(self) -> Iterator[MemoryRecord]:
        return iter(sorted(self._records.values(), key=lambda r: r.id))

    def count(self) -> int:
        return len(self._records)

    def add(self, text: str, score: float = 0.0, metadata: Optional[Dict[str, float]] = None) -> MemoryRecord:
        record = MemoryRecord(id=self._next_id, text=text, score=score, metadata=metadata or {})
        self._records[self._next_id] = record
        self._next_id += 1
        return record

    def update(self, record_id: int, **fields: float | str) -> MemoryRecord:
        record = self._records.get(record_id)
        if record is None:
            raise KeyError(f"{self._name} record {record_id} not found")
        data = record.__dict__.copy()
        data.update(fields)
        updated = MemoryRecord(**data)
        self._records[record_id] = updated
        return updated

    def remove(self, record_id: int) -> None:
        if record_id in self._records:
            del self._records[record_id]

    def clear(self) -> None:
        self._records.clear()

    def snapshot(self) -> List[MemoryRecord]:
        return list(self.iter())

    def filter(self, predicate: Callable[[MemoryRecord], bool]) -> List[MemoryRecord]:
        return [record for record in self._records.values() if predicate(record)]

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            self._name: [
                {
                    "id": record.id,
                    "text": record.text,
                    "score": record.score,
                    "metadata": record.metadata,
                }
                for record in self.iter()
            ]
        }


@dataclass
class StructuredMemory:
    """Grouping of the three canonical tables used across modules."""

    facts: MemoryTable = field(default_factory=lambda: MemoryTable("facts"))
    hypotheses: MemoryTable = field(default_factory=lambda: MemoryTable("hypotheses"))
    todos: MemoryTable = field(default_factory=lambda: MemoryTable("todos"))

    def reset(self) -> None:
        self.facts.clear()
        self.hypotheses.clear()
        self.todos.clear()

    def ingest_trace(self, trace: Dict[str, Sequence[str]]) -> None:
        for text in trace.get("facts", []):
            self.facts.add(text)
        for text in trace.get("hypotheses", []):
            self.hypotheses.add(text)
        for text in trace.get("todos", []):
            self.todos.add(text)

    def serialize(self) -> Dict[str, List[Dict[str, object]]]:
        data: Dict[str, List[Dict[str, object]]] = {}
        data.update(self.facts.to_dict())
        data.update(self.hypotheses.to_dict())
        data.update(self.todos.to_dict())
        return data

    def as_runtime_mapping(self) -> Dict[str, object]:
        return {
            "facts": [record.__dict__ for record in self.facts.iter()],
            "hypotheses": [record.__dict__ for record in self.hypotheses.iter()],
            "todos": [record.__dict__ for record in self.todos.iter()],
        }
