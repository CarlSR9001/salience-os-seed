"""Introspection interface exposing runtime internals for reflection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from ..controller.actions import ControllerAction
from ..controller.policy import SalienceControllerPolicy
from ..meta.state import MetaState
from ..operators.verifier import VerificationOutcome, VerifierSuite
from ..memory import StructuredMemory
from .workspace import WorkspaceViewer


@dataclass
class ControllerScore:
    action: ControllerAction
    score: float
    rationale: str


class IntrospectionInterface:
    """Read-only view into runtime internals for reflection."""

    def __init__(
        self,
        controller: SalienceControllerPolicy,
        verifier: VerifierSuite,
        memory: StructuredMemory,
        meta_state: MetaState,
        workspace: Optional[WorkspaceViewer] = None,
    ) -> None:
        self._controller = controller
        self._verifier = verifier
        self._memory = memory
        self._meta_state = meta_state
        self._workspace = workspace
        self._last_salience: Mapping[str, float] = {}
        self._last_memory_snapshot: Mapping[str, object] = {}
        self._last_verifications: List[VerificationOutcome] = []
        self._last_controller_scores: List[ControllerScore] = []
        self._last_yearning_snapshot: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------
    def update_salience(self, salience_map: Mapping[str, float]) -> None:
        self._last_salience = dict(salience_map)

    def update_memory_snapshot(self, snapshot: Mapping[str, object]) -> None:
        self._last_memory_snapshot = dict(snapshot)

    def record_verification(self, outcome: VerificationOutcome) -> None:
        self._last_verifications.append(outcome)
        if len(self._last_verifications) > 32:
            self._last_verifications.pop(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_salience_vector(self) -> Dict[str, float]:
        return dict(self._last_salience)

    def get_controller_scores(self) -> List[ControllerScore]:
        diagnostics = self._controller.last_scores()
        scores = [ControllerScore(action=a, score=s, rationale=r) for a, s, r in diagnostics]
        if scores:
            self._last_controller_scores = scores
        return scores or list(self._last_controller_scores)

    def get_meta_trajectory(self, steps: int = 10) -> List[MetaState]:
        history = self._meta_state.history
        if not history:
            return []
        return history[-steps:]

    def get_memory_diff(self) -> Dict[str, object]:
        current = self._memory.as_runtime_mapping()
        diff: Dict[str, object] = {"added": {}, "removed": {}, "changed": {}}
        for key, value in current.items():
            if key not in self._last_memory_snapshot:
                diff["added"][key] = value
            elif self._last_memory_snapshot[key] != value:
                diff["changed"][key] = {"before": self._last_memory_snapshot[key], "after": value}
        for key, value in self._last_memory_snapshot.items():
            if key not in current:
                diff["removed"][key] = value
        return diff

    def get_verification_log(self, n: int = 5) -> List[VerificationOutcome]:
        return self._last_verifications[-n:]

    def get_policy_stats(self) -> Dict[str, object]:
        return self._controller.policy_diagnostics()

    def get_yearning_state(self, refresh: bool = True) -> Dict[str, Dict[str, float]]:
        if refresh:
            snapshot = self._controller.s_prime.snapshot_yearnings()
            if snapshot:
                self._last_yearning_snapshot = snapshot
        return dict(self._last_yearning_snapshot)

    def adjust_controller_dynamics(
        self,
        updates: Mapping[str, float],
        *,
        max_step: float = 0.1,
    ) -> Dict[str, float]:
        applied = self._controller.s_prime.guarded_adjust_dynamics(updates, max_step=max_step)
        if applied:
            self._last_yearning_snapshot = self._controller.s_prime.snapshot_yearnings()
        return applied

    def get_workspace_listing(self, relative_path: str = ".") -> List[Dict[str, object]]:
        if not self._workspace:
            return []
        entries = self._workspace.list_entries(relative_path)
        return [
            {
                "path": str(entry.path),
                "is_dir": entry.is_dir,
                "size": entry.size,
            }
            for entry in entries
        ]

    def read_workspace_file(self, relative_path: str, max_chars: Optional[int] = 2000) -> str:
        if not self._workspace:
            raise RuntimeError("Workspace viewer not configured")
        return self._workspace.read_text(relative_path, max_chars=max_chars)

    def search_workspace(self, keyword: str, max_results: int = 10) -> List[Dict[str, object]]:
        if not self._workspace:
            return []
        return self._workspace.search(keyword, max_results=max_results)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def render_reflection_context(self) -> Dict[str, object]:
        return {
            "salience": self.get_salience_vector(),
            "controller_scores": [
                {
                    "action": score.action,
                    "score": score.score,
                    "rationale": score.rationale,
                }
                for score in self.get_controller_scores()
            ],
            "meta": {
                "current": self._meta_state.snapshot(),
                "history": [state.snapshot() for state in self.get_meta_trajectory(steps=5)],
            },
            "memory_diff": self.get_memory_diff(),
            "verification": [outcome.evidence for outcome in self.get_verification_log(5)],
            "policy": self.get_policy_stats(),
            "yearning": self.get_yearning_state(refresh=True),
        }

    def attach_salience(self, salience_map: Mapping[str, float]) -> None:
        self.update_salience(salience_map)

    def attach_memory_snapshot(self, snapshot: Mapping[str, object]) -> None:
        self.update_memory_snapshot(snapshot)

    def attach_verification(self, outcome: VerificationOutcome) -> None:
        self.record_verification(outcome)

    def request_reflection(self, boost: float = 1.0) -> None:
        self._controller.enqueue_reflection(boost=max(0.0, float(boost)))
