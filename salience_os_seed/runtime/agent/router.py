"""Routing utilities coordinating local and MCP tool execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from .session import AgentSession, ToolDescriptor, ToolOrigin


@dataclass(slots=True)
class ToolRequest:
    """Request describing a single tool invocation."""

    name: str
    payload: Mapping[str, object] | None = None
    descriptor: ToolDescriptor | None = None


@dataclass(slots=True)
class ToolBatch:
    """Batch of tool requests scheduled for execution together."""

    descriptor: ToolDescriptor
    requests: Sequence[ToolRequest]
    max_concurrency: int


class ToolRouter:
    """Route tool invocations across local runtime and MCP clients."""

    def __init__(
        self,
        session: AgentSession,
        *,
        max_cpu_parallelism: int = 2,
        remote_batch_size: int = 4,
    ) -> None:
        self._session = session
        self._max_cpu_parallelism = max(1, max_cpu_parallelism)
        self._remote_batch_size = max(1, remote_batch_size)

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def max_cpu_parallelism(self) -> int:
        return self._max_cpu_parallelism

    def enumerate_tools(self) -> Sequence[ToolDescriptor]:
        """Expose the current tool descriptors known to the router."""

        return self._session.available_tools()

    def plan(self, requests: Iterable[ToolRequest]) -> Sequence[ToolBatch]:
        """Build an execution plan with CPU-friendly batching semantics."""

        resolved_requests = [self._resolve_request(request) for request in requests]
        grouped: MutableMapping[str, List[ToolRequest]] = defaultdict(list)
        descriptor_index: MutableMapping[str, ToolDescriptor] = {}
        for request in resolved_requests:
            descriptor = request.descriptor
            if descriptor is None:
                continue
            grouped[descriptor.name].append(request)
            descriptor_index[descriptor.name] = descriptor
        batches: List[ToolBatch] = []
        for name, bucket in grouped.items():
            descriptor = descriptor_index[name]
            if descriptor.origin is ToolOrigin.MCP:
                batches.extend(self._chunk_requests(descriptor, bucket, self._remote_batch_size))
                continue
            if descriptor.cpu_bound:
                batches.extend(self._chunk_requests(descriptor, bucket, 1))
            else:
                batches.extend(
                    self._chunk_requests(
                        descriptor,
                        bucket,
                        min(self._max_cpu_parallelism, len(bucket)),
                    )
                )
        return batches

    def _resolve_request(self, request: ToolRequest) -> ToolRequest:
        if request.descriptor and request.descriptor.name == request.name:
            return request
        descriptor = self._session.describe_tool(request.name)
        return ToolRequest(name=request.name, payload=request.payload, descriptor=descriptor)

    @staticmethod
    def _chunk_requests(
        descriptor: ToolDescriptor,
        requests: Sequence[ToolRequest],
        batch_size: int,
    ) -> Sequence[ToolBatch]:
        batches: List[ToolBatch] = []
        if batch_size <= 0:
            batch_size = 1
        for index in range(0, len(requests), batch_size):
            slice_requests = requests[index : index + batch_size]
            batches.append(
                ToolBatch(
                    descriptor=descriptor,
                    requests=list(slice_requests),
                    max_concurrency=batch_size if descriptor.origin is ToolOrigin.MCP else min(batch_size, len(slice_requests)),
                )
            )
        return batches


__all__ = ["ToolBatch", "ToolRequest", "ToolRouter"]
