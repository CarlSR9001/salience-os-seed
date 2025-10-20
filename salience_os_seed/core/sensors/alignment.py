"""AIM alignment sensor measuring goal adherence.

The sensor consumes embedding vectors for the current prompt/task state and the
explicit goal description. The runtime is responsible for providing embeddings
(e.g., via a lightweight encoder or cached vectors). We compute a cosine
similarity and optionally smooth the output to reduce jitter.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .base import MedianMADNormalizer, Sensor


class AlignmentSensor(Sensor):
    """Estimate AIM (alignment with the stated goal).

    The runtime should expose `state["embeddings"]["prompt"]` and
    `memory["goal"]["embedding"]` as 1D float arrays. The cosine similarity ranges
    between -1 and 1; we rescale to `[0, 1]` before normalisation.
    """

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        smoothing_window: int = 8,
    ) -> None:
        super().__init__(name="alignment", domain="alignment", normaliser=normaliser)
        if smoothing_window < 4:
            raise ValueError("smoothing_window must be >= 4 for alignment sensor")
        self._window = smoothing_window
        self._buffer: Deque[float] = deque(maxlen=smoothing_window)
        self._cached_goal: Tuple[Tuple[float, ...], np.ndarray | None] | None = None
        self._cached_projector_sig: Tuple[int, int, float] | None = None
        self._last_distance_source: str = "direct"

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        prompt_vec = _extract_vector(state, ("embeddings", "prompt"))
        goal_vec = _extract_vector(memory, ("goal", "embedding"))
        if prompt_vec is None or goal_vec is None:
            return 0.0
        projector = _extract_projection(meta)
        projector_sig = _projector_signature(projector)

        projected_prompt = _project_vector(prompt_vec, projector)

        goal_key = _vector_fingerprint(goal_vec, projector_sig)
        if (
            self._cached_goal is None
            or self._cached_goal[0] != goal_key
            or self._cached_projector_sig != projector_sig
        ):
            projected_goal = _project_vector(goal_vec, projector)
            goal_unit = _unit_vector(projected_goal)
            self._cached_goal = (goal_key, goal_unit)
            self._cached_projector_sig = projector_sig
            self._last_distance_source = "recomputed"
        else:
            goal_unit = self._cached_goal[1]
            self._last_distance_source = "cached"

        prompt_unit = _unit_vector(projected_prompt)
        if prompt_unit is None or goal_unit is None:
            return 0.0
        similarity = float(np.dot(prompt_unit, goal_unit))
        # rescale from [-1, 1] to [0, 1]
        scaled = 0.5 * (similarity + 1.0)
        self._buffer.append(scaled)
        if len(self._buffer) < 4:
            return scaled
        return float(np.mean(self._buffer))

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        buffer_min = float(min(self._buffer)) if self._buffer else raw_value
        buffer_max = float(max(self._buffer)) if self._buffer else raw_value
        projection_dim = 0
        if self._cached_projector_sig is not None:
            projection_dim = self._cached_projector_sig[0]
        return {
            "raw_alignment": raw_value,
            "window_min": buffer_min,
            "window_max": buffer_max,
            "normalised": norm_value,
            "projection_dim": float(projection_dim),
            "goal_cache": 1.0 if self._last_distance_source == "cached" else 0.0,
        }


def _extract_vector(source: Mapping[str, object], path: Sequence[str]) -> np.ndarray | None:
    cursor: object = source
    for key in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    if cursor is None:
        return None
    arr = np.asarray(cursor, dtype=np.float64)
    if arr.ndim != 1:
        return None
    return arr
def _extract_projection(meta: Mapping[str, object]) -> np.ndarray | None:
    candidate = meta.get("salience_projection") or meta.get("alignment_projection")
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


def _unit_vector(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return None
    return vector / norm


def _projector_signature(projector: np.ndarray | None) -> Tuple[int, int, float] | None:
    if projector is None:
        return None
    return (int(projector.shape[0]), int(projector.shape[1]), float(projector.sum()))


def _vector_fingerprint(vector: np.ndarray, projector_sig: Tuple[int, int, float] | None) -> Tuple[float, ...]:
    mean = float(np.mean(vector))
    std = float(np.std(vector))
    var = std * std
    length = float(vector.shape[0])
    projector_hash = 0.0 if projector_sig is None else float(sum(projector_sig))
    return (length, mean, var, projector_hash)
