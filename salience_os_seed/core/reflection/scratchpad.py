"""Scratchpad working memory for chain-of-thought reasoning."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Mapping, MutableSequence, Optional, Sequence

import numpy as np


@dataclass
class ScratchpadTrace:
    """A committed reasoning trace with metadata for retrieval."""

    steps: Sequence[str]
    outcome: bool
    token_count: int
    embedding: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def summary(self, max_chars: int = 120) -> str:
        joined = " ".join(self.steps)
        return joined[:max_chars] + ("…" if len(joined) > max_chars else "")


class Scratchpad:
    """Token-budgeted working memory for reasoning traces."""

    def __init__(self, max_tokens: int = 512, history_capacity: int = 128) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.history_capacity = history_capacity
        self.buffer: Deque[ScratchpadTrace] = deque(maxlen=history_capacity)
        self.current_trace: MutableSequence[str] = []
        self.trace_history: List[ScratchpadTrace] = []
        self._current_tokens = 0
        self._token_history: Deque[int] = deque(maxlen=history_capacity)

    # ---------------------------------------------------------------------
    # Trace lifecycle
    # ---------------------------------------------------------------------
    def append(self, thought: str) -> None:
        """Append a reasoning step to the current trace, respecting budget."""

        if not isinstance(thought, str):
            raise TypeError("thought must be a string")
        tokens = self._estimate_tokens(thought)
        if self._current_tokens + tokens > self.max_tokens:
            # Best-effort: drop oldest steps until budget satisfied
            self._trim_current(tokens)
            if self._current_tokens + tokens > self.max_tokens:
                # Still cannot fit; discard the incoming thought
                return
        self.current_trace.append(thought)
        self._current_tokens += tokens

    def commit(self, outcome: bool, metadata: Optional[Mapping[str, object]] = None) -> Optional[ScratchpadTrace]:
        """Persist the current trace and reset working buffer."""

        if not self.current_trace:
            return None
        steps = tuple(self.current_trace)
        token_count = self._current_tokens
        embedding = self._encode_trace(steps)
        trace = ScratchpadTrace(
            steps=steps,
            outcome=bool(outcome),
            token_count=token_count,
            embedding=embedding,
            metadata=dict(metadata or {}),
        )
        self.buffer.append(trace)
        self._token_history.append(token_count)
        if len(self.buffer) > self.buffer.maxlen:
            self.buffer.popleft()
        if len(self._token_history) > self._token_history.maxlen:
            self._token_history.popleft()
        self.trace_history.append(trace)
        self.current_trace.clear()
        self._current_tokens = 0
        return trace

    def reset(self) -> None:
        """Clear current trace without committing."""

        self.current_trace.clear()
        self._current_tokens = 0

    # ------------------------------------------------------------------
    # Retrieval & analytics
    # ------------------------------------------------------------------
    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 3) -> List[ScratchpadTrace]:
        """Return the top-k traces most similar to the query embedding."""

        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be 1-D")
        if not self.buffer:
            return []
        similarities = []
        q_norm = np.linalg.norm(query_embedding) + 1e-9
        for trace in self.buffer:
            t_norm = np.linalg.norm(trace.embedding) + 1e-9
            sim = float(np.dot(query_embedding, trace.embedding) / (q_norm * t_norm))
            similarities.append((sim, trace))
        similarities.sort(key=lambda item: item[0], reverse=True)
        return [trace for _, trace in similarities[: top_k]]

    def summarize(self, max_traces: int = 3) -> str:
        """Produce a human-readable summary of recent reasoning."""

        if not self.buffer:
            return "<empty>"
        items = list(self.buffer)[-max_traces:]
        parts = []
        for trace in items:
            outcome = "✓" if trace.outcome else "✗"
            parts.append(f"{outcome} {trace.summary(80)}")
        return " | ".join(parts)

    def rolling_token_mean(self) -> float:
        if not self._token_history:
            return 0.0
        return float(np.mean(self._token_history))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _trim_current(self, incoming_tokens: int) -> None:
        """Drop earliest steps until budget can accommodate incoming tokens."""

        while self.current_trace and self._current_tokens + incoming_tokens > self.max_tokens:
            removed = self.current_trace.pop(0)
            self._current_tokens -= self._estimate_tokens(removed)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough heuristic: whitespace tokens with guard for empty strings.
        return max(1, len(text.strip().split()))

    def _encode_trace(self, steps: Sequence[str]) -> np.ndarray:
        """Encode a trace into a deterministic embedding via hashed n-grams."""

        counts: Counter[str] = Counter()
        for step in steps:
            tokens = step.lower().split()
            for token in tokens:
                counts[token] += 1
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + "_" + tokens[i + 1]
                counts[bigram] += 1
        if not counts:
            return np.zeros(128, dtype=np.float32)
        vector = np.zeros(128, dtype=np.float32)
        for key, value in counts.items():
            idx = hash(key) % 128
            vector[idx] += float(value)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def export_traces(self) -> List[ScratchpadTrace]:
        """Return a copy of committed traces for persistence."""

        return list(self.buffer)

    def import_traces(self, traces: Iterable[ScratchpadTrace]) -> None:
        """Restore traces from storage, respecting capacity."""

        for trace in traces:
            self.buffer.append(trace)
            self._token_history.append(trace.token_count)
        while len(self.buffer) > self.buffer.maxlen:
            self.buffer.popleft()
        while len(self._token_history) > self._token_history.maxlen:
            self._token_history.popleft()
