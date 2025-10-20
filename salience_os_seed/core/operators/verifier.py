"""Verifier suite orchestrating calculators, retrieval, and skeptic checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

from .memory_ops import MemoryOperator
from .scratchpad_verifier import evaluate_scratchpad_trace


@dataclass
class VerificationOutcome:
    """Container for verifier results."""

    passed: bool
    evidence: Dict[str, object]


class VerifierSuite:
    """Coordinate multiple lightweight verifiers.

    The suite exposes an interface returning whether verification passed and any
    supporting evidence (calculator results, retrieval hits, skeptic notes).
    """

    def __init__(self) -> None:
        self.calculators: Dict[str, Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]] = {}
        self.retrievers: Dict[str, Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]] = {}
        self.skeptics: Dict[str, Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]] = {}
        self._register_default_checks()

    def register_calculator(
        self, name: str, fn: Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]
    ) -> None:
        self.calculators[name] = fn

    def register_retriever(
        self, name: str, fn: Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]
    ) -> None:
        self.retrievers[name] = fn

    def register_skeptic(
        self, name: str, fn: Callable[[Mapping[str, object]], Tuple[bool, Dict[str, object]]]
    ) -> None:
        self.skeptics[name] = fn

    def run(self, context: Mapping[str, object]) -> VerificationOutcome:
        evidence: Dict[str, object] = {}
        passed = True
        for name, calculator in self.calculators.items():
            ok, payload = calculator(context)
            evidence[f"calc::{name}"] = payload
            passed &= ok
        for name, retriever in self.retrievers.items():
            ok, payload = retriever(context)
            evidence[f"retrieval::{name}"] = payload
            passed &= ok
        for name, skeptic in self.skeptics.items():
            ok, payload = skeptic(context)
            evidence[f"skeptic::{name}"] = payload
            passed &= ok
        return VerificationOutcome(passed=passed, evidence=evidence)

    # ------------------------------------------------------------------
    # Built-in checks
    # ------------------------------------------------------------------
    def _register_default_checks(self) -> None:
        self.register_skeptic("scratchpad", self._scratchpad_check)

    def _scratchpad_check(self, context: Mapping[str, object]) -> Tuple[bool, Dict[str, object]]:
        trace = context.get("scratchpad") or []
        if isinstance(trace, str):
            trace = [trace]
        if not isinstance(trace, (list, tuple)):
            trace = []
        memory = context.get("memory_snapshot")
        if not isinstance(memory, Mapping):
            memory = {}
        return evaluate_scratchpad_trace(trace, memory)
