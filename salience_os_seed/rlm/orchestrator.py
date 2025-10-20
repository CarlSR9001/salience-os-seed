"""Recursive language model orchestrator leveraging SalienceRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.controller.actions import ControllerAction, ControllerOperator, ControllerPatch
from ..core.reflection import WorkspaceViewer
from ..runtime.orchestrator import SalienceRuntime
from .model_client import ModelClient
from .policy import RLMPolicy
from .store import RLMStore
from .tools import build_toolkit, tool_descriptors
from .types import LLMResponse, Prompt, PromptMessage, RLNode, RLMTraceEvent, RunContext, ToolInvocation


@dataclass
class RLMResult:
    answer: Optional[str]
    confidence: float
    trace: Sequence[RLMTraceEvent]
    budget_used: int


class RLM:
    """Run recursive long-context reasoning backed by the salience runtime."""

    def __init__(
        self,
        model: ModelClient,
        runtime: SalienceRuntime,
        workspace: WorkspaceViewer,
        policy: Optional[RLMPolicy] = None,
        scratch_root: Optional[str] = None,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.policy = policy or RLMPolicy()
        self.store = RLMStore(workspace, scratch_root=scratch_root)
        self.toolkit = tuple(build_toolkit(self.store))
        self._tool_map = {tool.name: tool for tool in self.toolkit}
        self._tool_specs = tool_descriptors(self.toolkit)

    def run(self, task: str, scope: Optional[Sequence[str]] = None) -> RLMResult:
        stack: List[RLNode] = [
            RLNode(
                task=task,
                scope=tuple(scope or self.store.root_scope()),
                depth=0,
                budget=self.policy.per_call_cap,
            )
        ]
        trace: List[RLMTraceEvent] = []
        global_budget = self.policy.total_budget
        best = {"answer": None, "confidence": 0.0}
        stale_rounds = 0

        while stack and global_budget > 0:
            node = stack.pop()
            call_budget = min(node.budget, self.policy.per_call_cap, global_budget)
            prompt = self._make_prompt(node, trace, global_budget)
            response = self.model.generate(prompt, tools=self._tool_specs, max_tokens=call_budget)
            global_budget -= response.tokens
            trace.append(self._log_response(node, response))

            ctx = RunContext(
                store=self.store,
                policy=self.policy,
                runtime=self.runtime,
                scratchpad=self.runtime.scratchpad,
                execution_meta={"node_depth": node.depth, "task": node.task},
            )

            new_evidence = False
            for call in response.tool_calls:
                result = self._dispatch_tool(call, ctx)
                trace.append(self._log_tool(call, result))
                if call.name == "spawn":
                    children = self._materialize_children(result, node)
                    if children:
                        new_evidence = True
                        stack.extend(children)
                else:
                    if result:
                        new_evidence = True

            if response.confidence is not None and response.confidence > best["confidence"]:
                best = {"answer": response.text.strip(), "confidence": float(response.confidence)}
            elif not best["answer"] and response.text.strip():
                best = {"answer": response.text.strip(), "confidence": best["confidence"]}

            stale_rounds = 0 if new_evidence else stale_rounds + 1
            if self._should_halt(best, stale_rounds):
                break

        budget_used = self.policy.total_budget - max(global_budget, 0)
        return RLMResult(
            answer=best["answer"],
            confidence=best["confidence"],
            trace=trace,
            budget_used=budget_used,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _make_prompt(self, node: RLNode, trace: Sequence[RLMTraceEvent], budget: int) -> Prompt:
        system_instructions = (
            "You are a recursive research assistant. Use the provided tools (read, scan, summarize, write, spawn) "
            "to gather evidence from the workspace. Only reference artifacts after reading them."
        )
        scope_description = "\n".join(self.store.describe_scope(node.scope)) or "(scope summaries unavailable)"
        trace_excerpt = self._recent_trace(trace, limit=4)
        user_block = (
            f"TASK: {node.task}\n"
            f"DEPTH: {node.depth}\n"
            f"AVAILABLE SCOPE IDS: {list(node.scope)}\n"
            f"SCOPE DETAILS:\n{scope_description}\n"
            f"REMAINING GLOBAL BUDGET: {budget}\n"
            "Respond with JSON including fields: answer, confidence (0-1), and optional tool_calls[] (name + arguments)."
        )
        if trace_excerpt:
            user_block += f"\nRECENT TRACE:\n{trace_excerpt}"
        messages = [
            PromptMessage(role="system", content=system_instructions),
            PromptMessage(role="user", content=user_block),
        ]
        metadata = {"node_depth": node.depth, "remaining_budget": budget}
        return Prompt(messages=messages, metadata=metadata)

    def _recent_trace(self, trace: Sequence[RLMTraceEvent], limit: int) -> str:
        if not trace:
            return ""
        tail = trace[-limit:]
        lines = []
        for event in tail:
            lines.append(f"- {event.kind}: {event.payload}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    def _dispatch_tool(self, call: ToolInvocation, ctx: RunContext) -> Any:
        tool = self._tool_map.get(call.name)
        if not tool:
            return {"error": f"unknown_tool::{call.name}"}
        try:
            return tool.handler(call.arguments, ctx)
        except Exception as exc:  # pragma: no cover - defensive guard
            return {"error": f"tool_failure::{call.name}", "detail": str(exc)}

    def _materialize_children(self, result: Mapping[str, Any], parent: RLNode) -> List[RLNode]:
        children: List[RLNode] = []
        raw_children = result.get("children", []) if isinstance(result, Mapping) else []
        for entry in raw_children:
            task = str(entry.get("task", parent.task))
            scope = tuple(self.store.clamp_scope(entry.get("scope")) or parent.scope)
            budget = int(entry.get("budget", parent.budget))
            budget = max(200, min(budget, self.policy.per_call_cap))
            salience_score = float(entry.get("salience", parent.parent_salience))
            child_depth = parent.depth + 1
            if not self.policy.allow_child(child_depth, salience_score):
                continue
            node = RLNode(task=task, scope=scope, depth=child_depth, budget=budget, parent_salience=salience_score)
            children.append(node)
        if not children:
            return []
        ranked = sorted(children, key=lambda item: self._score_child(item), reverse=True)
        return list(ranked[: self.policy.clamp_children(len(ranked))])

    def _score_child(self, node: RLNode) -> float:
        salience = max(0.0, min(1.0, node.parent_salience)) if node.parent_salience else 0.4
        action = ControllerAction(
            cot_depth=min(max(node.depth, 1), 3),
            operator=ControllerOperator.REFLECT,
            patch=ControllerPatch.NONE,
        )
        salience_map = {
            "novelty": salience,
            "alignment": 0.55 + 0.35 * salience,
            "progress": 0.35,
            "coherence": 0.6,
            "drag": max(0.0, 0.4 - 0.25 * salience),
            "cost": 0.1 + 0.2 * node.depth,
        }
        score, _ = self.runtime.controller.s_prime.score_action(
            action,
            salience_map,
            self.runtime.meta_state.snapshot(),
            cooldown_time=self.runtime.controller.state.cooldown_remaining,
        )
        return float(score)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_response(self, node: RLNode, response: LLMResponse) -> RLMTraceEvent:
        payload = {
            "task": node.task,
            "depth": node.depth,
            "text": response.text,
            "tokens": response.tokens,
            "confidence": response.confidence,
        }
        return RLMTraceEvent(kind="model", payload=payload)

    def _log_tool(self, call: ToolInvocation, result: Any) -> RLMTraceEvent:
        payload = {
            "tool": call.name,
            "args": dict(call.arguments),
            "result": self._summarize_result(result),
        }
        return RLMTraceEvent(kind="tool", payload=payload)

    @staticmethod
    def _summarize_result(result: Any) -> Any:
        if isinstance(result, str) and len(result) > 320:
            return result[:317] + "…"
        if isinstance(result, Mapping):
            trimmed = {}
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 160:
                    trimmed[key] = value[:157] + "…"
                else:
                    trimmed[key] = value
            return trimmed
        return result

    def _should_halt(self, best: Mapping[str, Any], stale_rounds: int) -> bool:
        if best.get("confidence", 0.0) >= self.policy.confidence_threshold:
            return True
        if stale_rounds >= self.policy.halt_no_new_evidence_rounds:
            return True
        return False
