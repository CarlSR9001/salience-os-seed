"""Validation for the recursive language model scaffold and toolchain."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from salience_os_seed.core.reflection import WorkspaceViewer
from salience_os_seed.runtime.orchestrator import SalienceRuntime
from salience_os_seed.rlm.model_client import ModelClient
from salience_os_seed.rlm.orchestrator import RLM
from salience_os_seed.rlm.policy import RLMPolicy
from salience_os_seed.rlm.types import LLMResponse, Prompt, PromptMessage, ToolInvocation


class DummyModel(ModelClient):
    """Deterministic model client for testing orchestration."""

    def __init__(self, script: Dict[str, Any]) -> None:
        self.script = script
        self.calls = 0

    def generate(self, prompt: Prompt, tools, max_tokens: int) -> LLMResponse:
        key = self.calls
        self.calls += 1
        entry = self.script.get(key)
        if entry is None:
            return LLMResponse(text="(no answer)", tokens=max_tokens)
        tool_calls = [ToolInvocation(name=item["name"], arguments=item.get("arguments", {})) for item in entry.get("tool_calls", [])]
        return LLMResponse(
            text=entry.get("text", ""),
            tokens=entry.get("tokens", max_tokens),
            tool_calls=tool_calls,
            confidence=entry.get("confidence"),
        )


@pytest.fixture
def workspace(tmp_path) -> WorkspaceViewer:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "doc_a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (root / "doc_b.txt").write_text("delta epsilon zeta", encoding="utf-8")
    return WorkspaceViewer(root)


def test_rlm_orchestrator_runs_and_respects_salience(workspace):
    runtime = SalienceRuntime()
    script = {
        0: {
            "text": "Investigating scope.",
            "tokens": 120,
            "tool_calls": [
                {"name": "scan", "arguments": {"query": "alpha"}},
                {
                    "name": "spawn",
                    "arguments": {
                        "task": "detail doc",
                        "children": [
                            {"task": "summ doc_a", "scope": ["doc_a.txt"], "salience": 0.6},
                            {"task": "summ doc_b", "scope": ["doc_b.txt"], "salience": 0.05},
                        ],
                    },
                },
            ],
        },
        1: {
            "text": "Doc A summary: alpha beta",
            "confidence": 0.8,
            "tokens": 110,
        },
    }
    model = DummyModel(script)
    rlm = RLM(model=model, runtime=runtime, workspace=workspace, policy=RLMPolicy(total_budget=2_000, per_call_cap=256))
    result = rlm.run("Investigate workspace")

    assert result.answer == "Doc A summary: alpha beta"
    assert pytest.approx(result.confidence, rel=1e-5) == 0.8
    assert result.budget_used <= 2_000
    spawned = [event for event in result.trace if event.kind == "tool" and event.payload.get("tool") == "spawn"]
    assert spawned, "spawn tool should have been invoked"
    child_entries = spawned[0].payload["result"].get("children", [])
    assert any(entry.get("task") == "summ doc_a" for entry in child_entries)
