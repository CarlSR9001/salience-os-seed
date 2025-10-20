"""Four-dimensional spatial reasoning utilities for Salience OS Seed."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Mapping, MutableSequence, Sequence

import numpy as np


def _hashed_floats(key: str, *, count: int) -> list[float]:
    """Generate a deterministic float sequence in ``[-1, 1]`` for ``key``."""

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    base = [(byte / 255.0) * 2.0 - 1.0 for byte in digest]
    repeats = int(math.ceil(count / len(base)))
    tiled: list[float] = (base * repeats)[:count]
    return [float(value) for value in tiled]


def _make_basis(length: int) -> list[list[float]]:
    """Construct a 4×``length`` deterministic projection basis without heavy numpy deps."""

    idx_range = [float(i) for i in range(length)]
    rows: list[list[float]] = []
    for offset, freq in [(0.0, 0.13), (0.7, 0.17), (1.4, 0.21), (2.1, 0.09)]:
        row = [
            math.sin(value * freq + offset) + math.cos(value * (freq * 0.67) + offset * 0.5)
            for value in idx_range
        ]
        norm = math.sqrt(sum(val * val for val in row)) or 1.0
        rows.append([val / norm for val in row])
    return rows


@dataclass(frozen=True)
class FourDCoordinate:
    """Point in a four-dimensional hypercube where ``w`` encodes ana/kata."""

    x: float
    y: float
    z: float
    w: float

    @property
    def ana(self) -> float:
        """Magnitude of the ana-facing component (``w`` >= 0)."""

        return max(self.w, 0.0)

    @property
    def kata(self) -> float:
        """Magnitude of the kata-facing component (``w`` < 0)."""

        return max(-self.w, 0.0)

    def distance_to(self, other: "FourDCoordinate") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
            + (self.w - other.w) ** 2
        )

    def blend(self, other: "FourDCoordinate", weight: float = 0.5) -> "FourDCoordinate":
        inv = 1.0 - weight
        return FourDCoordinate(
            x=self.x * inv + other.x * weight,
            y=self.y * inv + other.y * weight,
            z=self.z * inv + other.z * weight,
            w=self.w * inv + other.w * weight,
        )

    def project_iso(self, yaw: float = 0.65, pitch: float = 0.52) -> tuple[float, float]:
        """Project the 4D point into a 2D plane for visualization."""

        # Rotate X/Y into the horizontal plane.
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rx = cos_yaw * self.x - sin_yaw * self.y
        ry = sin_yaw * self.x + cos_yaw * self.y

        # Treat Z as height and W as colour/intensity; fold W into height via pitch.
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        rz = cos_pitch * self.z - sin_pitch * self.w
        screen_x = rx
        screen_y = ry * 0.5 + rz
        return screen_x, screen_y

    def to_dict(self) -> Mapping[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "w": float(self.w),
            "ana": float(self.ana),
            "kata": float(self.kata),
        }


@dataclass
class FourDPath:
    """Ordered collection of ``FourDCoordinate`` instances with metadata."""

    coordinates: MutableSequence[FourDCoordinate] = field(default_factory=list)
    labels: MutableSequence[str] = field(default_factory=list)

    def append(self, coord: FourDCoordinate, label: str = "") -> None:
        self.coordinates.append(coord)
        self.labels.append(label)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.coordinates)

    def centroid(self) -> FourDCoordinate:
        if not self.coordinates:
            return FourDCoordinate(0.0, 0.0, 0.0, 0.0)
        total_x = sum(coord.x for coord in self.coordinates)
        total_y = sum(coord.y for coord in self.coordinates)
        total_z = sum(coord.z for coord in self.coordinates)
        total_w = sum(coord.w for coord in self.coordinates)
        count = len(self.coordinates)
        return FourDCoordinate(
            total_x / count,
            total_y / count,
            total_z / count,
            total_w / count,
        )

    def total_displacement(self) -> float:
        if len(self.coordinates) < 2:
            return 0.0
        total = 0.0
        for left, right in zip(self.coordinates[:-1], self.coordinates[1:]):
            total += left.distance_to(right)
        return total

    def project_points(self) -> List[Mapping[str, float]]:
        projected = []
        for idx, coord in enumerate(self.coordinates):
            px, py = coord.project_iso()
            projected.append(
                {
                    "index": idx,
                    "u": float(px),
                    "v": float(py),
                    "w": float(coord.w),
                }
            )
        return projected

    def ascii_projection(self, width: int = 36, height: int = 14) -> str:
        if not self.coordinates:
            return "<empty>"
        projected = self.project_points()
        xs = [point["u"] for point in projected]
        ys = [point["v"] for point in projected]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        grid = [[" "] * width for _ in range(height)]
        for idx, point in enumerate(projected):
            col = int((point["u"] - min_x) / span_x * (width - 1))
            row = int((point["v"] - min_y) / span_y * (height - 1))
            row = max(0, min(height - 1, height - 1 - row))
            col = max(0, min(width - 1, col))
            glyph = str((idx + 1) % 10)
            grid[row][col] = glyph
        return "\n".join("".join(row) for row in grid)

    def summary(self) -> str:
        steps = len(self.coordinates)
        displacement = self.total_displacement()
        centroid = self.centroid()
        return (
            f"{steps} steps | drift={displacement:.2f} | "
            f"centroid=({centroid.x:.2f},{centroid.y:.2f},{centroid.z:.2f}|w={centroid.w:.2f})"
        )

    def to_dict(self) -> Mapping[str, object]:
        centroid = self.centroid()
        return {
            "points": [
                {
                    **coord.to_dict(),
                    "label": label,
                    "projection": proj,
                }
                for coord, label, proj in zip(
                    self.coordinates, self.labels, self.project_points()
                )
            ],
            "centroid": centroid.to_dict(),
            "summary": self.summary(),
        }


def embedding_to_coordinate(embedding: np.ndarray) -> FourDCoordinate:
    """Project an embedding vector into the canonical 4D manifold."""

    if hasattr(embedding, "tolist"):
        vector_list = embedding.tolist()
    else:
        vector_list = list(embedding)
    vector = [float(value) for value in vector_list]
    if not vector:
        return FourDCoordinate(0.0, 0.0, 0.0, 0.0)
    basis = _make_basis(len(vector))
    coords = [
        sum(row[i] * vector[i] for i in range(len(row)))
        for row in basis
    ]
    norm = math.sqrt(sum(value * value for value in coords)) or 1.0
    coords = [value / norm for value in coords]
    return FourDCoordinate(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))


def text_to_coordinate(text: str) -> FourDCoordinate:
    """Embed ``text`` via hashed bag-of-ngrams and project to 4D."""

    tokens = [tok for tok in text.lower().split() if tok]
    if not tokens:
        return FourDCoordinate(0.0, 0.0, 0.0, 0.0)
    length = 64
    vec = [0.0] * length
    for token in tokens:
        hashed = _hashed_floats(token, count=length)
        vec = [v + h for v, h in zip(vec, hashed)]
    for idx in range(len(tokens) - 1):
        bigram = tokens[idx] + "_" + tokens[idx + 1]
        hashed = _hashed_floats(bigram, count=length)
        vec = [v + 0.5 * h for v, h in zip(vec, hashed)]
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 1e-9:
        vec = [value / norm for value in vec]
    return embedding_to_coordinate(vec)


def trace_to_path(steps: Sequence[str]) -> FourDPath:
    """Create a smoothed ``FourDPath`` from textual reasoning steps."""

    path = FourDPath()
    previous = FourDCoordinate(0.0, 0.0, 0.0, 0.0)
    for step in steps:
        coord = text_to_coordinate(step)
        blended = previous.blend(coord, weight=0.65)
        path.append(blended, step)
        previous = blended
    return path


def payload_to_coordinate(payload: Mapping[str, object]) -> FourDCoordinate:
    """Encode a parameter payload dictionary into a 4D coordinate."""

    if not payload:
        return FourDCoordinate(0.0, 0.0, 0.0, 0.0)
    items = [f"{key}::{repr(value)}" for key, value in sorted(payload.items())]
    combined = " | ".join(items)
    return text_to_coordinate(combined)


__all__ = [
    "FourDCoordinate",
    "FourDPath",
    "embedding_to_coordinate",
    "text_to_coordinate",
    "trace_to_path",
    "payload_to_coordinate",
]
