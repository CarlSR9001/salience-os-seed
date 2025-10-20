"""Episodic memory utilities for SalienceOS Seed."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence


@dataclass
class Episode:
    """Summary of a runtime episode."""

    episode_id: str
    task_type: str
    salience_profile: Sequence[float]
    actions_taken: Sequence[str]
    outcome: Optional[bool]
    scratchpad_summary: str
    lessons_learned: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class EpisodicStore:
    """In-memory buffer with optional persistence for episodes."""

    def __init__(self, path: Optional[Path] = None, max_size: int = 512) -> None:
        self._path = path
        self._max_size = max(1, max_size)
        self._episodes: List[Episode] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def record_episode(self, episode: Episode) -> Episode:
        self._episodes.append(episode)
        if len(self._episodes) > self._max_size:
            self._episodes.pop(0)
        self._save()
        return episode

    def retrieve_similar(self, salience_profile: Sequence[float], top_k: int = 3) -> List[Episode]:
        if not self._episodes:
            return []
        key = list(salience_profile)
        similarities = []
        for episode in self._episodes:
            similarity = _cosine_similarity(key, list(episode.salience_profile))
            similarities.append((similarity, episode))
        similarities.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in similarities[:top_k]]

    def summarize_lessons(self, limit: int = 5) -> str:
        if not self._episodes:
            return "<no episodes>"
        recent = self._episodes[-limit:]
        return " | ".join(f"{ep.task_type}:{ep.lessons_learned}" for ep in recent)

    def to_dict(self) -> List[Mapping[str, object]]:
        return [episode.__dict__ for episode in self._episodes]

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    episode = Episode(**payload)
                    self._episodes.append(episode)
        except (json.JSONDecodeError, OSError):
            # Corrupt file; start fresh but keep the file for future writes.
            self._episodes = []

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("w", encoding="utf-8") as handle:
                for episode in self._episodes:
                    handle.write(json.dumps(episode.__dict__, ensure_ascii=False) + "\n")
        except OSError:
            pass


def build_episode(
    salience_profile: Mapping[str, float],
    actions: Sequence[str],
    outcome: Optional[bool],
    scratchpad_summary: str,
    lessons: str,
    task_type: str = "generic",
    metadata: Optional[Mapping[str, object]] = None,
) -> Episode:
    episode_id = f"ep-{int(time.time() * 1000)}"
    profile = [float(value) for _, value in sorted(salience_profile.items())]
    return Episode(
        episode_id=episode_id,
        task_type=task_type,
        salience_profile=profile,
        actions_taken=list(actions),
        outcome=outcome,
        scratchpad_summary=scratchpad_summary,
        lessons_learned=lessons,
        metadata=dict(metadata or {}),
    )


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
