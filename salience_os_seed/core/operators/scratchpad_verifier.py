"""Scratchpad verification heuristics for reflective reasoning traces."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence, Tuple


def evaluate_scratchpad_trace(
    trace: Sequence[str],
    memory_snapshot: Mapping[str, object],
    min_coherence: float = 0.35,
    min_utility: float = 0.2,
) -> Tuple[bool, Mapping[str, float]]:
    """Score a reasoning trace for coherence, grounding, and utility.

    Parameters
    ----------
    trace:
        Ordered list of scratchpad thoughts.
    memory_snapshot:
        Structured memory mapping (facts/hypotheses/todos) used to assess grounding.
    min_coherence:
        Threshold for coherence score to be considered acceptable.
    min_utility:
        Threshold for utility score (information per token).
    """

    if not trace:
        return False, {"coherence": 0.0, "grounding": 0.0, "utility": 0.0, "length": 0.0}

    connectors = {"because", "therefore", "thus", "however", "but", "so", "since"}
    connector_hits = 0
    total_tokens = 0
    unique_tokens: Counter[str] = Counter()

    for step in trace:
        tokens = [token.strip(".,!?").lower() for token in step.split() if token]
        total_tokens += len(tokens)
        unique_tokens.update(tokens)
        if any(token in connectors for token in tokens):
            connector_hits += 1

    coherence_score = connector_hits / max(1, len(trace))
    utility_score = min(1.0, len(unique_tokens) / max(1, total_tokens + len(trace)))
    grounding_score = _estimate_grounding(trace, memory_snapshot)

    passed = coherence_score >= min_coherence and utility_score >= min_utility
    evidence = {
        "coherence": round(coherence_score, 3),
        "grounding": round(grounding_score, 3),
        "utility": round(utility_score, 3),
        "length": float(len(trace)),
    }
    return passed, evidence


def _estimate_grounding(trace: Sequence[str], memory_snapshot: Mapping[str, object]) -> float:
    facts = _collect_memory_strings(memory_snapshot, "facts")
    hypotheses = _collect_memory_strings(memory_snapshot, "hypotheses")
    references = 0
    if not facts and not hypotheses:
        return 0.0
    for step in trace:
        lower = step.lower()
        if any(fragment in lower for fragment in facts):
            references += 1
        elif any(fragment in lower for fragment in hypotheses):
            references += 0.5
    max_refs = max(1, len(trace))
    return min(1.0, references / max_refs)


def _collect_memory_strings(memory_snapshot: Mapping[str, object], key: str) -> Sequence[str]:
    entries = memory_snapshot.get(key, [])
    results = []
    for entry in entries:
        if isinstance(entry, Mapping):
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                results.append(text.lower())
        elif isinstance(entry, str) and entry.strip():
            results.append(entry.lower())
    return results
