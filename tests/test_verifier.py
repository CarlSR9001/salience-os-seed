"""Tests for scratchpad verification heuristics."""

from __future__ import annotations

from salience_os_seed.core.operators.scratchpad_verifier import evaluate_scratchpad_trace


def test_scratchpad_verifier_passes_coherent_trace():
    trace = [
        "Break the task into parts because it is complex.",
        "Therefore compute the first sub-result.",
        "Finally combine the components to reach the answer.",
    ]
    memory = {
        "facts": [{"text": "component"}],
    }
    ok, evidence = evaluate_scratchpad_trace(trace, memory)
    assert ok
    assert evidence["coherence"] > 0.3
    assert evidence["grounding"] > 0.0
    assert evidence["utility"] > 0.2


def test_scratchpad_verifier_flags_low_utility_trace():
    trace = ["hmm", "hmm", "hmm"]
    memory = {}
    ok, evidence = evaluate_scratchpad_trace(trace, memory)
    assert not ok
    assert evidence["utility"] <= 0.2


def test_scratchpad_verifier_handles_empty_trace():
    ok, evidence = evaluate_scratchpad_trace([], {})
    assert not ok
    assert evidence["coherence"] == 0.0
    assert evidence["utility"] == 0.0
    assert evidence["length"] == 0.0
