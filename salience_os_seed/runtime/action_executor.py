"""Action execution strategies for the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional

import torch

from ..core.controller import (
    BanditTrainer,
    ControllerAction,
    ControllerDecision,
    ControllerOperator,
    ControllerPatch,
)
from ..core.memory import StructuredMemory
from ..core.operators import GraphReasoner, MemoryOperator, SparseJumpTeleporter, VerifierSuite, SASSCore
from ..core.reflection import IntrospectionInterface


@dataclass(slots=True)
class ActionContext:
    """Mutable state shared across action handlers."""

    hidden_states: Optional[torch.Tensor] = None
    layer_states: Optional[list[torch.Tensor]] = None
    training_active: bool = False


class BaseActionHandler:
    """Interface for action handlers."""

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        raise NotImplementedError


@dataclass(slots=True)
class VerifyActionHandler(BaseActionHandler):
    verifier: VerifierSuite
    bandit: BanditTrainer
    introspection: IntrospectionInterface
    memory: StructuredMemory
    lambda_cost: float

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        context = dict(state)
        context.setdefault("memory_snapshot", self.memory.as_runtime_mapping())
        outcome = self.verifier.run(context)
        reward = (1.0 if outcome.passed else -1.0) - self.lambda_cost * float(salience.get("cost", 0.0))
        self.bandit.update(decision.action, reward)
        self.introspection.attach_verification(outcome)
        return outcome.passed


@dataclass(slots=True)
class SassActionHandler(BaseActionHandler):
    sass: SASSCore
    teleporter: SparseJumpTeleporter
    graph_reasoner: GraphReasoner
    memory: StructuredMemory
    bandit: BanditTrainer
    context: ActionContext
    lambda_cost: float

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        action = decision.action
        hidden_states = state.get("hidden_states") or self.context.hidden_states
        if hidden_states is None:
            raise ValueError("SASS execution requires hidden states in the runtime state")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        layer_states = state.get("layer_states") or self.context.layer_states
        hyper_deltas = state.get("hyper_deltas")
        detach_states = state.get("detach_states")
        if not isinstance(detach_states, bool):
            detach_states = not self.sass.training
        output, new_layer_states = self.sass(
            hidden_states,
            layer_states,
            hyper_deltas,
            detach_states=detach_states,
        )
        sequence_id = int(state.get("sequence_id", 0))
        if action.operator is ControllerOperator.SASS_WITH_JUMP:
            trigger = bool(
                state.get("teleport_trigger")
                or (
                    float(salience.get("novelty", 0.0)) > 1.0
                    and float(salience.get("alignment", 0.0)) > 0.8
                )
            )
            residual = self.teleporter(output, sequence_id=sequence_id, trigger=trigger)
            output[:, -1, :] = output[:, -1, :] + residual
            if float(salience.get("progress", 0.0)) > 0.8 or state.get("reasoner_trigger"):
                output, _ = self.graph_reasoner(
                    output,
                    self.memory,
                    entity_hints=state.get("entity_hints", ()),
                )
        self.context.layer_states = new_layer_states
        if state.get("training_active") is not None:
            self.context.training_active = bool(state.get("training_active"))
        if self.context.training_active:
            self.context.hidden_states = output
        else:
            self.context.hidden_states = output.detach()
        reward = float(salience.get("progress", 0.0)) - self.lambda_cost * float(salience.get("cost", 0.0))
        self.bandit.update(action, reward)
        return None


@dataclass(slots=True)
class MemoryActionHandler(BaseActionHandler):
    memory_operator: MemoryOperator
    bandit: BanditTrainer

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        verb = state.get("memory_verb")
        applied = False
        if isinstance(verb, Mapping):
            result = self.memory_operator.execute(verb)
            applied = bool(result.applied)
        reward = (1.0 if applied else -0.2) - 0.1 * float(salience.get("drag", 0.0))
        self.bandit.update(decision.action, reward)
        return None


@dataclass(slots=True)
class ToolActionHandler(BaseActionHandler):
    bandit: BanditTrainer
    tool_adapter: "ToolInvocationAdapter"

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        invoked = self.tool_adapter.invoke(decision.action, state)
        reward = 0.5 if invoked else -0.1
        self.bandit.update(decision.action, reward)
        return None


@dataclass(slots=True)
class ReflectionActionHandler(BaseActionHandler):
    reflection: "ReflectionManager"
    bandit: BanditTrainer

    def handle(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        reward = self.reflection.reflect(decision.action, state, salience)
        self.bandit.update(decision.action, reward)
        return None


@dataclass(slots=True)
class ActionExecutor:
    verify_handler: VerifyActionHandler
    sass_handler: SassActionHandler
    memory_handler: MemoryActionHandler
    tool_handler: ToolActionHandler
    reflection_handler: ReflectionActionHandler

    def execute(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        operator = decision.action.operator
        if operator is ControllerOperator.VERIFY:
            return self.verify_handler.handle(decision, state, salience)
        if operator in {ControllerOperator.SASS, ControllerOperator.SASS_WITH_JUMP}:
            return self.sass_handler.handle(decision, state, salience)
        if operator is ControllerOperator.MEMORY_OP:
            return self.memory_handler.handle(decision, state, salience)
        if operator is ControllerOperator.TOOL:
            return self.tool_handler.handle(decision, state, salience)
        if operator is ControllerOperator.REFLECT:
            return self.reflection_handler.handle(decision, state, salience)
        return None


class ToolInvocationAdapter:
    """Adapter allowing tool invocation via direct call or MCP."""

    def __init__(
        self,
        runtime_tools: MutableMapping[str, object] | None = None,
        operator_tool_map: Mapping[ControllerOperator, object] | None = None,
    ) -> None:
        self.runtime_tools = runtime_tools or {}
        self.operator_tool_map = dict(operator_tool_map or {})
        self.mcp_session: "MCPToolSession | None" = None

    def register_mcp_session(self, session: "MCPToolSession") -> None:
        self.mcp_session = session

    def invoke(self, action: ControllerAction, state: Mapping[str, object]) -> bool:
        candidate = self._resolve_candidate(action, state)
        if callable(candidate):
            candidate(state)
            return True
        if not candidate:
            return False

        tool_name = candidate
        tools = state.get("tools")
        if isinstance(tools, Mapping):
            tool_fn = tools.get(tool_name)
            if callable(tool_fn):
                tool_fn(state)
                return True
        if self.mcp_session and self.mcp_session.invoke(tool_name, state):
            return True
        runtime_fn = self.runtime_tools.get(tool_name)
        if callable(runtime_fn):
            runtime_fn(state)
            return True
        return False

    def _resolve_candidate(
        self, action: ControllerAction, state: Mapping[str, object]
    ) -> object | None:
        explicit = state.get("tool_name")
        if isinstance(explicit, str) and explicit:
            return explicit

        mapped = self.operator_tool_map.get(action.operator)
        resolved = self._resolve_mapped_tool(mapped, action)
        if resolved:
            return resolved

        if action.patch is ControllerPatch.NONE:
            return None

        fallback = action.patch.name.lower()
        if self._tool_exists(fallback, state):
            return fallback
        return None

    def _resolve_mapped_tool(self, mapped: object | None, action: ControllerAction) -> object | None:
        if mapped is None:
            return None
        if isinstance(mapped, Mapping):
            patch_tool = mapped.get(action.patch)
            if patch_tool:
                return patch_tool
            return mapped.get(ControllerPatch.NONE)
        return mapped

    def _tool_exists(self, tool_name: str, state: Mapping[str, object]) -> bool:
        tools = state.get("tools")
        if isinstance(tools, Mapping) and callable(tools.get(tool_name)):
            return True
        runtime_fn = self.runtime_tools.get(tool_name)
        return callable(runtime_fn)


class MCPToolSession:
    """Minimal MCP-compatible shim used by the runtime."""

    def __init__(self, client) -> None:
        self.client = client

    def invoke(self, tool_name: str, state: Mapping[str, object]) -> bool:
        try:
            self.client.call_tool(tool_name, state)
            return True
        except Exception:
            return False


__all__ = [
    "ActionContext",
    "ActionExecutor",
    "VerifyActionHandler",
    "SassActionHandler",
    "MemoryActionHandler",
    "ToolActionHandler",
    "ReflectionActionHandler",
    "ToolInvocationAdapter",
    "MCPToolSession",
]
