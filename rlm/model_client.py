"""Model client abstractions for the recursive language model orchestrator."""

from __future__ import annotations

import abc
from typing import Iterable, Optional

from .types import LLMResponse, Prompt, ToolInvocation


class ModelClient(abc.ABC):
    """Interface that wraps a language model endpoint."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: Prompt,
        tools: Iterable[dict],
        max_tokens: int,
    ) -> LLMResponse:
        """Produce a response given the prompt and available tools."""

    def subcall(
        self,
        prompt: Prompt,
        tools: Iterable[dict],
        max_tokens: int,
        policy: Optional[dict] = None,
    ) -> LLMResponse:
        """Optional convenience for recursive calls; defaults to `generate`."""

        return self.generate(prompt=prompt, tools=tools, max_tokens=max_tokens)
