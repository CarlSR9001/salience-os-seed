"""Policy configuration for the recursive language model orchestrator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLMPolicy:
    """Execution guardrails for recursive language modelling.

    Defaults favour deliberate research-style workflows rather than chatty
    back-and-forth. Budgets assume a medium context LLM running with external
    memory; adjust to match deployment constraints.
    """

    total_budget: int = 40_000  # aggregate prompt tokens across the run
    per_call_cap: int = 1_536  # generous cap for planning turns
    max_depth: int = 4
    max_children: int = 5
    halt_no_new_evidence_rounds: int = 3
    confidence_threshold: float = 0.78
    salience_prune_threshold: float = 0.15

    def allow_child(self, depth: int, salience_score: float) -> bool:
        if depth > self.max_depth:
            return False
        return salience_score >= self.salience_prune_threshold

    def clamp_children(self, count: int) -> int:
        return min(count, self.max_children)
