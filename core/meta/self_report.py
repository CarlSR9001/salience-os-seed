"""Self-report surface for the meta-state.

This module provides a compact textual summary of the current meta-state vector
that downstream UIs or logging sinks can display. Reports are intended to be
machine- and human-readable (one short line, key=value pairs).
"""

from __future__ import annotations

from typing import Mapping


def render_self_report(meta_snapshot: Mapping[str, float]) -> str:
    """Render a single-line summary of the meta-state."""

    confidence = meta_snapshot.get("confidence", 0.0)
    difficulty = meta_snapshot.get("difficulty", 0.0)
    blind_spot = meta_snapshot.get("blind_spot", 0.0)
    roi = meta_snapshot.get("roi", 0.0)
    return (
        f"confidence={confidence:+.2f} "
        f"difficulty={difficulty:.2f} "
        f"blind_spot={blind_spot:+.2f} "
        f"roi={roi:+.2f}"
    )
