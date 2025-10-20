"""Utilities for loading chain-of-thought curriculum lessons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

REQUIRED_KEYS = {"task", "salience_context", "reasoning_trace", "answer", "meta_lesson"}


@dataclass(frozen=True)
class CurriculumExample:
    task: str
    salience_context: Mapping[str, float]
    reasoning_trace: Sequence[str]
    answer: str
    meta_lesson: str
    path: Path


def load_lesson(path: str | Path) -> List[CurriculumExample]:
    lesson_path = Path(path)
    examples: List[CurriculumExample] = []
    with lesson_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - surfaced in tests
                raise ValueError(f"Invalid JSONL entry in {lesson_path} line {line_number}: {exc}") from exc
            missing = REQUIRED_KEYS - payload.keys()
            if missing:
                raise ValueError(f"Missing keys {missing} in {lesson_path} line {line_number}")
            example = CurriculumExample(
                task=str(payload["task"]),
                salience_context=_coerce_float_mapping(payload["salience_context"]),
                reasoning_trace=_coerce_trace(payload["reasoning_trace"]),
                answer=str(payload["answer"]),
                meta_lesson=str(payload["meta_lesson"]),
                path=lesson_path,
            )
            examples.append(example)
    return examples


def load_curriculum(root: str | Path) -> Dict[str, List[CurriculumExample]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Curriculum root '{root_path}' does not exist")
    lessons: Dict[str, List[CurriculumExample]] = {}
    for jsonl_path in root_path.rglob("*.jsonl"):
        relative = jsonl_path.relative_to(root_path)
        lessons[str(relative)] = load_lesson(jsonl_path)
    return lessons


def iter_examples(root: str | Path) -> Iterator[CurriculumExample]:
    curriculum = load_curriculum(root)
    for examples in curriculum.values():
        yield from examples


def _coerce_float_mapping(payload: Mapping[str, object]) -> Mapping[str, float]:
    return {key: float(value) for key, value in payload.items()}


def _coerce_trace(payload: Iterable[object]) -> List[str]:
    trace = []
    for step in payload:
        if not isinstance(step, str):
            raise ValueError("Reasoning trace must contain only strings")
        trace.append(step)
    return trace
