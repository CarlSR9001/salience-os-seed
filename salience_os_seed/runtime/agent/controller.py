"""High-level controller orchestrating tool execution with MCP streaming."""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Deque, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

from .router import ToolBatch, ToolRequest, ToolRouter
from .session import AgentSession, ToolOrigin

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    """Result emitted from executing a tool request."""

    request: ToolRequest
    output: object | None


class AgentController:
    """Coordinate tool execution while handling streaming MCP responses."""

    def __init__(
        self,
        session: AgentSession,
        router: ToolRouter,
        *,
        followup_factory: Callable[[ToolRequest, object], Sequence[ToolRequest]] | None = None,
        stream_workers: int | None = None,
    ) -> None:
        self._session = session
        self._router = router
        self._followup_factory = followup_factory
        self._followups: Deque[ToolRequest] = deque()
        self._followup_lock = Lock()
        max_workers = stream_workers or max(2, router.max_cpu_parallelism)
        self._stream_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mcp-stream")
        self._pending_streams: set[Future[None]] = set()

    def close(self) -> None:
        """Shutdown controller resources."""

        for future in list(self._pending_streams):
            future.cancel()
        self._stream_executor.shutdown(wait=False, cancel_futures=True)

    def dispatch(self, requests: Iterable[ToolRequest], state: Mapping[str, object]) -> Sequence[ToolResult]:
        """Execute tool requests, handling MCP streaming responses lazily."""

        combined_requests = list(requests)
        combined_requests.extend(self._drain_followups())
        if not combined_requests:
            return []
        plan = self._router.plan(combined_requests)
        results: List[ToolResult] = []
        for batch in plan:
            if batch.descriptor.origin is ToolOrigin.MCP:
                results.extend(self._execute_remote_batch(batch, state))
            else:
                results.extend(self._execute_local_batch(batch, state))
        return results

    # Execution helpers ------------------------------------------------

    def _execute_local_batch(
        self, batch: ToolBatch, base_state: Mapping[str, object]
    ) -> Sequence[ToolResult]:
        results: List[ToolResult] = []
        for request in batch.requests:
            descriptor = batch.descriptor
            entrypoint = descriptor.entrypoint
            if not callable(entrypoint):
                logger.debug("Skipping local tool %s without callable entrypoint", descriptor.name)
                continue
            state = self._merge_state(base_state, request.payload)
            try:
                output = entrypoint(state)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Local tool %s failed", descriptor.name)
                output = None
            results.append(ToolResult(request=request, output=output))
        return results

    def _execute_remote_batch(
        self, batch: ToolBatch, base_state: Mapping[str, object]
    ) -> Sequence[ToolResult]:
        results: List[ToolResult] = []
        for request in batch.requests:
            state = self._merge_state(base_state, request.payload)
            output = self._session.call_remote_tool(request.name, state)
            if self._maybe_stream(request, output, base_state):
                continue
            results.append(ToolResult(request=request, output=output))
        return results

    def _maybe_stream(
        self, request: ToolRequest, output: object | None, base_state: Mapping[str, object]
    ) -> bool:
        if output is None:
            return False
        if isinstance(output, (str, bytes, Mapping)):
            return False
        if isinstance(output, Iterable):
            future = self._stream_executor.submit(
                self._consume_stream,
                request,
                iter(output),
                dict(base_state),
            )
            self._pending_streams.add(future)
            future.add_done_callback(self._pending_streams.discard)
            return True
        return False

    def _consume_stream(
        self,
        request: ToolRequest,
        stream: Iterator[object],
        state_snapshot: MutableMapping[str, object],
    ) -> None:
        for chunk in stream:
            try:
                self._handle_stream_chunk(request, chunk, state_snapshot)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Error while processing MCP stream for %s", request.name)
                break

    def _handle_stream_chunk(
        self,
        request: ToolRequest,
        chunk: object,
        state_snapshot: MutableMapping[str, object],
    ) -> None:
        if not self._followup_factory:
            return
        followups = self._followup_factory(request, chunk)
        if not followups:
            return
        with self._followup_lock:
            self._followups.extend(followups)

    def _drain_followups(self) -> Sequence[ToolRequest]:
        with self._followup_lock:
            drained = list(self._followups)
            self._followups.clear()
        return drained

    @staticmethod
    def _merge_state(
        base_state: Mapping[str, object], payload: Mapping[str, object] | None
    ) -> MutableMapping[str, object]:
        if not payload:
            return dict(base_state)
        merged: MutableMapping[str, object] = dict(base_state)
        merged.update(payload)
        return merged


__all__ = ["AgentController", "ToolResult"]
