"""Coherence sensor measuring continuity across reasoning signals."""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, MutableMapping, Optional

import numpy as np

from .base import MedianMADNormalizer, Sensor


class CoherenceSensor(Sensor):
    """Track semantic, structural, and logical continuity in recent steps."""

    def __init__(self, normaliser: MedianMADNormalizer, window: int = 10) -> None:
        super().__init__(name="coherence", domain="coherence", normaliser=normaliser)
        if window < 2:
            raise ValueError("CoherenceSensor window must be >= 2")
        self._embedding_history: Deque[np.ndarray] = deque(maxlen=window)
        self._action_history: Deque[str] = deque(maxlen=window)
        self._scratchpad_history: Deque[str] = deque(maxlen=window)
        self._last_components: Mapping[str, float] = {}

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        semantic = np.clip(self._semantic_continuity(state, meta), 0.0, 1.0)
        structural = np.clip(self._structural_continuity(state, meta), 0.0, 1.0)
        logical = np.clip(self._logical_consistency(state, memory, meta), 0.0, 1.0)
        scratchpad = np.clip(self._scratchpad_coherence(state, meta), 0.0, 1.0)
        score = 0.3 * semantic + 0.25 * structural + 0.25 * logical + 0.2 * scratchpad
        self._last_components = {
            "semantic": float(semantic),
            "structural": float(structural),
            "logical": float(logical),
            "scratchpad": float(scratchpad),
        }
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Sub-measures
    # ------------------------------------------------------------------
    def _semantic_continuity(self, state: Mapping[str, object], meta: Mapping[str, object]) -> float:
        embedding = self._extract_embedding(state, meta)
        if embedding is None:
            return 1.0
        if not self._embedding_history:
            self._embedding_history.append(embedding)
            return 1.0
        sims = [self._cosine_similarity(embedding, past) for past in self._embedding_history]
        self._embedding_history.append(embedding)
        return float(np.clip(np.mean(sims), -1.0, 1.0)) * 0.5 + 0.5

    def _structural_continuity(self, state: Mapping[str, object], meta: Mapping[str, object]) -> float:
        action = str(state.get("last_action", ""))
        if action:
            self._action_history.append(action)
        if len(self._action_history) < 3:
            return 1.0
        entropy_hint = meta.get("action_transition_entropy")
        if isinstance(entropy_hint, (int, float)):
            entropy_val = float(np.clip(entropy_hint, 0.0, 1.0))
            return float(np.clip(1.0 - entropy_val, 0.0, 1.0))
        switches = sum(
            1 for idx in range(len(self._action_history) - 1)
            if self._action_history[idx] != self._action_history[idx + 1]
        )
        return max(0.0, 1.0 - switches / max(1, len(self._action_history) - 1))

    def _logical_consistency(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        contradictions = state.get("contradictions", 0)
        if isinstance(contradictions, (int, float)):
            contradictions = max(0.0, float(contradictions))
        else:
            contradictions = 0.0
        penalty = min(1.0, contradictions / 3.0)
        meta_penalty = meta.get("reasoning_conflict")
        if isinstance(meta_penalty, (int, float)):
            penalty = max(penalty, float(np.clip(meta_penalty, 0.0, 1.0)))
        return max(0.0, 1.0 - penalty)

    def _scratchpad_coherence(self, state: Mapping[str, object], meta: Mapping[str, object]) -> float:
        trace_text = state.get("scratchpad_text")
        if not isinstance(trace_text, str) or not trace_text.strip():
            return 1.0
        meta_hint = meta.get("scratchpad_consistency")
        if isinstance(meta_hint, (int, float)):
            hint_val = float(np.clip(meta_hint, 0.0, 1.0))
        else:
            hint_val = None
        tokens = trace_text.split()
        connector_count = sum(1 for token in tokens if token.lower() in {"because", "therefore", "thus", "however", "but"})
        diversity = len(set(tokens)) / max(1, len(tokens))
        heuristic = 0.4 * min(1.0, connector_count / 3.0) + 0.6 * np.clip(diversity, 0.0, 1.0)
        score = heuristic if hint_val is None else 0.5 * heuristic + 0.5 * hint_val
        self._scratchpad_history.append(trace_text[:512])
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _extract_embedding(self, state: Mapping[str, object], meta: Mapping[str, object]) -> Optional[np.ndarray]:
        projector = _extract_projection(meta)
        embedding = state.get("embedding")
        if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
            vector = embedding.astype(np.float64, copy=False)
            return _unit_vector(_project_vector(vector, projector))
        hidden = state.get("hidden_states")
        if isinstance(hidden, np.ndarray) and hidden.ndim >= 1:
            vector = hidden.flatten()[-256:].astype(np.float64, copy=False)
            return _unit_vector(_project_vector(vector, projector))
        embed_list = state.get("embedding")
        if isinstance(embed_list, (list, tuple)) and embed_list:
            arr = np.asarray(embed_list, dtype=np.float64)
            if arr.ndim == 1:
                vector = arr[-256:]
                return _unit_vector(_project_vector(vector, projector))
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        payload: MutableMapping[str, float] = {
            "raw_coherence": raw_value,
            "normalised": norm_value,
        }
        for key, value in self._last_components.items():
            payload[f"component_{key}"] = float(value)
        return payload


def _extract_projection(meta: Mapping[str, object]) -> np.ndarray | None:
    candidate = meta.get("salience_projection") or meta.get("coherence_projection")
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        candidate = candidate.get("matrix")
    arr = np.asarray(candidate, dtype=np.float64)
    if arr.ndim != 2:
        return None
    return arr


def _project_vector(vector: np.ndarray, projector: np.ndarray | None) -> np.ndarray:
    if projector is None:
        return vector
    return projector @ vector


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return None
    return vector / norm
