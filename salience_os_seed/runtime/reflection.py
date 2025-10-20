"""Reflection facilities decoupled from the runtime orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..core.controller import ControllerAction
from ..core.meta import MetaState
from ..core.reflection import PatternLibrary, Scratchpad, WorkspaceViewer
from ..core.reflection import IntrospectionInterface


@dataclass(slots=True)
class ReflectionManager:
    """Coordinates scratchpad, pattern library, and introspection."""

    scratchpad: Scratchpad
    pattern_library: PatternLibrary
    introspection: IntrospectionInterface
    workspace_viewer: WorkspaceViewer | None
    meta_state: MetaState

    def reflect(
        self,
        action: ControllerAction,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> float:
        sorted_salience: Sequence[tuple[str, float]] = sorted(
            ((key, float(value)) for key, value in salience.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_salience = ", ".join(f"{key}={value:.2f}" for key, value in sorted_salience[:5])
        self.scratchpad.append(f"step {state.get('step_index', 0)} reflection: {top_salience}")
        patterns = self.pattern_library.retrieve(salience)
        if not patterns and self.pattern_library.patterns:
            patterns = self.pattern_library.patterns[:1]
        for pattern in patterns:
            self.scratchpad.append(f"pattern::{pattern.name} -> {pattern.description}")
        meta_snapshot = self.meta_state.snapshot()
        self.scratchpad.append(
            "meta state: confidence={:.2f} roi={:.2f}".format(
                float(meta_snapshot.get("confidence", 0.0)),
                float(meta_snapshot.get("roi", 0.0)),
            )
        )
        if self.workspace_viewer:
            listing = self.introspection.get_workspace_listing(".")
            if listing:
                focus_paths = ", ".join(item["path"] for item in listing[:3])
                if focus_paths:
                    self.scratchpad.append(f"workspace focus: {focus_paths}")
        context_snippet = state.get("context_snippet")
        if isinstance(context_snippet, str) and context_snippet.strip():
            self.scratchpad.append(f"context noted: {context_snippet.strip()[:160]}")
        metadata = {
            "salience": {key: value for key, value in sorted_salience[:6]},
            "patterns": [pattern.name for pattern in patterns],
        }
        self.scratchpad.commit(outcome=True, metadata=metadata)
        for pattern in patterns:
            self.pattern_library.log_usage(
                pattern,
                success=True,
                benefit=float(salience.get("progress", 0.0)),
                cost=float(state.get("token_cost", 1.0)),
            )
        reward = (
            0.3 * float(salience.get("novelty", 0.0))
            + 0.2 * float(salience.get("uncertainty", 0.0))
            + 0.1 * float(salience.get("progress", 0.0))
            - 0.15 * float(salience.get("drag", 0.0))
        )
        return reward


__all__ = ["ReflectionManager"]
