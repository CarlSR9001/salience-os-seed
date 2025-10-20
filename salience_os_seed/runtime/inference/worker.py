"""Inference worker with MCP hint consultation."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .cache import HintCache


class InferenceWorker:
    """Run inference with optional MCP hint consultation."""

    def __init__(
        self,
        decoder: Callable[[Sequence[str] | str], Any],
        *,
        mcp_client: Any | None = None,
        hint_cache: HintCache | None = None,
        hint_timeout: float = 0.25,
    ) -> None:
        self._decoder = decoder
        self._mcp_client = mcp_client
        self._hint_cache = hint_cache or HintCache()
        self._hint_timeout = max(0.0, float(hint_timeout))

    async def generate(
        self,
        prompt_segments: Sequence[str] | str,
        *,
        context_key: str,
        mcp_payload: Mapping[str, Any] | None = None,
    ) -> Any:
        """Generate a decode result after consulting MCP for hints."""

        prompt = self._hint_cache.merge_prompt(prompt_segments, context_key)

        hints = await self._consult_mcp(context_key, mcp_payload or {})
        if hints is not None:
            self._hint_cache.store_hint(
                context_key, hints["hints"], version=hints.get("version")
            )
            prompt = self._hint_cache.merge_prompt(prompt_segments, context_key)

        return await self._decode(prompt)

    async def _decode(self, prompt: Sequence[str] | str) -> Any:
        """Execute the decoder, supporting both sync and async callables."""

        result = self._decoder(prompt)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _consult_mcp(
        self, context_key: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Ask the MCP bridge for hints, guarding with a timeout."""

        if self._mcp_client is None:
            return None
        if not hasattr(self._mcp_client, "fetch_hints"):
            return None

        request_payload: MutableMapping[str, Any] = dict(payload)
        request_payload.setdefault("context_key", context_key)

        try:
            result = self._mcp_client.fetch_hints(request_payload)
        except Exception:
            return None

        if inspect.isawaitable(result):
            try:
                response = await asyncio.wait_for(result, timeout=self._hint_timeout)
            except asyncio.TimeoutError:
                return None
        else:
            response = result

        return self._normalize_hint_response(response)

    def _normalize_hint_response(self, response: Any) -> Mapping[str, Any] | None:
        """Convert a raw MCP response into a cache-friendly mapping."""

        if not response:
            return None

        version = None
        hints_payload: Sequence[str] | str | None = None
        if isinstance(response, Mapping):
            version = response.get("version")
            hints_payload = response.get("hints") or response.get("hint")
        elif isinstance(response, (list, tuple)):
            hints_payload = response
        elif isinstance(response, str):
            hints_payload = [response]
        else:
            return None

        hints = self._normalize_hints(hints_payload)
        if not hints:
            return None

        try:
            version_value = int(version)
        except (TypeError, ValueError):
            version_value = self._hint_cache.required_version
        return {"version": version_value, "hints": hints}

    def _normalize_hints(
        self, payload: Sequence[str] | str | None
    ) -> tuple[str, ...]:
        if payload is None:
            return tuple()
        if isinstance(payload, str):
            candidate = [payload]
        else:
            candidate = payload

        normalized = tuple(str(item).strip() for item in candidate)
        normalized = tuple(filter(None, normalized))
        return normalized


__all__ = ["InferenceWorker"]
