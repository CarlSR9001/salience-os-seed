"""Core type definitions for the recursive language model scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


@dataclass
class PromptMessage:
    role: str
    content: str


@dataclass
class Prompt:
    messages: Sequence[PromptMessage]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ToolInvocation:
    """Structured tool call emitted by the model."""

    name: str
    arguments: Mapping[str, Any]


@dataclass
class LLMResponse:
    """Normalised response returned by the model client."""

    text: str
    tokens: int
    tool_calls: Sequence[ToolInvocation] = field(default_factory=tuple)
    confidence: Optional[float] = None


@dataclass
class ToolSpec:
    """Host-exposed tool contract."""

    name: str
    description: str
    schema: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any], "RunContext"], Any]


@dataclass
class RLNode:
    """Node in the recursive task graph."""

    task: str
    scope: Sequence[str]
    depth: int
    budget: int
    parent_salience: float = 0.0


@dataclass
class RLMTraceEvent:
    """Audit trail entries captured during execution."""

    kind: str
    payload: Mapping[str, Any]


@dataclass
class RunContext:
    """Runtime context passed to tool handlers."""

    store: "RLMStore"
    policy: "RLMPolicy"
    runtime: "SalienceRuntime"
    scratchpad: "Scratchpad"
    execution_meta: MutableMapping[str, Any]


# Forward references for type checking
if False:  # pragma: no cover
    from .policy import RLMPolicy
    from .store import RLMStore
    from ..runtime.orchestrator import SalienceRuntime
    from ..core.reflection.scratchpad import Scratchpad
