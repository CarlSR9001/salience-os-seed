"""Agent session management for local and MCP-backed tools."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
    cast,
)

from ..action_executor import MCPToolSession, ToolInvocationAdapter

logger = logging.getLogger(__name__)


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Protocol describing the MCP client surface consumed by the runtime."""

    name: str

    def call_tool(self, tool_name: str, state: Mapping[str, object]) -> object | None:
        """Invoke a tool exposed by the MCP client."""

    def list_tools(self) -> Sequence[Mapping[str, object]] | Sequence[str]:
        """Return an iterable description of tools made available by the client."""

    def close(self) -> None:
        """Tear down the client connection."""

    def connect(self) -> None:  # pragma: no cover - optional protocol hook
        """Optional hook establishing the client connection."""


class ToolOrigin(str, Enum):
    """Enumerate the possible sources for a tool binding."""

    LOCAL = "local"
    MCP = "mcp"


@dataclass(slots=True)
class ToolDescriptor:
    """Describes a tool available to the agent runtime."""

    name: str
    origin: ToolOrigin
    entrypoint: Optional[Callable[[Mapping[str, object]], object]] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def cpu_bound(self) -> bool:
        """Whether the tool should be treated as CPU intensive."""

        cpu_flag = self.metadata.get("cpu_bound")
        if isinstance(cpu_flag, bool):
            return cpu_flag
        return False


class _AggregatedMCPToolSession:
    """Composite MCP session delegating to multiple clients."""

    def __init__(self, clients: Sequence[MCPClientProtocol]) -> None:
        self._clients = list(clients)
        self._tool_clients: Dict[str, MCPClientProtocol] = {}

    def update_tool_cache(self, cache: Mapping[str, MCPClientProtocol]) -> None:
        self._tool_clients = dict(cache)

    def call(self, tool_name: str, state: Mapping[str, object]) -> object | None:
        client = self._tool_clients.get(tool_name)
        if client is None:
            return None
        try:
            return client.call_tool(tool_name, state)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("MCP client failed for tool %s", tool_name, exc_info=True)
            return None

    def invoke(self, tool_name: str, state: Mapping[str, object]) -> bool:
        return self.call(tool_name, state) is not None


class AgentSession:
    """Manage local tool bindings and MCP clients within an agent session."""

    def __init__(
        self,
        tool_adapter: ToolInvocationAdapter,
        *,
        local_tools: MutableMapping[str, Callable[[Mapping[str, object]], object]] | None = None,
        mcp_client_factories: Iterable[Callable[[], MCPClientProtocol | ExitStack]] | None = None,
    ) -> None:
        self.tool_adapter = tool_adapter
        self._local_tools = local_tools if local_tools is not None else tool_adapter.runtime_tools
        self._local_metadata: Dict[str, Mapping[str, object]] = {}
        self._client_factories = list(mcp_client_factories or [])
        self._exit_stack = ExitStack()
        self._mcp_clients: list[MCPClientProtocol] = []
        self._mcp_session: _AggregatedMCPToolSession | None = None
        self._tool_cache: "OrderedDict[str, ToolDescriptor]" = OrderedDict()

    def __enter__(self) -> "AgentSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def register_mcp_factory(self, factory: Callable[[], MCPClientProtocol | ExitStack]) -> None:
        """Register an MCP client factory for the session."""

        self._client_factories.append(factory)

    def bind_local_tool(
        self,
        name: str,
        entrypoint: Callable[[Mapping[str, object]], object],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Bind a callable tool to the local runtime."""

        self._local_tools[name] = entrypoint
        if metadata:
            self._local_metadata[name] = dict(metadata)
        self._refresh_tool_cache()

    def start(self) -> None:
        """Instantiate MCP clients and prepare tool bindings."""

        self._open_mcp_clients()
        self._refresh_tool_cache()

    def stop(self) -> None:
        """Tear down MCP clients and unregister remote tools."""

        if self.tool_adapter.mcp_session is self._mcp_session:
            self.tool_adapter.mcp_session = None
        self._mcp_session = None
        self._tool_cache.clear()
        self._mcp_clients.clear()
        try:
            self._exit_stack.close()
        finally:
            self._exit_stack = ExitStack()

    def available_tools(self) -> Sequence[ToolDescriptor]:
        """Return all tools available within the session."""

        self._refresh_tool_cache()
        return list(self._tool_cache.values())

    def describe_tool(self, name: str) -> ToolDescriptor | None:
        """Return the descriptor for a specific tool if available."""

        self._refresh_tool_cache()
        return self._tool_cache.get(name)

    def call_remote_tool(self, tool_name: str, state: Mapping[str, object]) -> object | None:
        """Invoke a remote MCP tool if the binding exists."""

        if not self._mcp_session:
            return None
        return self._mcp_session.call(tool_name, state)

    # Internal helpers -------------------------------------------------

    def _open_mcp_clients(self) -> None:
        clients: list[MCPClientProtocol] = []
        for factory in self._client_factories:
            client = self._instantiate_client(factory)
            if client is None:
                continue
            connect = getattr(client, "connect", None)
            if callable(connect):
                try:
                    connect()
                except Exception:  # pragma: no cover - defensive guard
                    logger.warning("Failed to connect MCP client %s", getattr(client, "name", "unknown"), exc_info=True)
                    continue
            clients.append(client)
        self._mcp_clients = clients
        if not clients:
            return
        self._mcp_session = _AggregatedMCPToolSession(clients)
        self.tool_adapter.register_mcp_session(self._mcp_session)  # type: ignore[arg-type]

    def _instantiate_client(
        self, factory: Callable[[], MCPClientProtocol | ExitStack]
    ) -> MCPClientProtocol | None:
        resource = factory()
        if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
            resource = self._exit_stack.enter_context(resource)  # type: ignore[arg-type]
        if isinstance(resource, MCPToolSession):  # pragma: no cover - legacy hook
            client = getattr(resource, "client", None)
            if client is None:
                return None
            resource = client
        missing = [
            attr
            for attr in ("call_tool", "list_tools")
            if not callable(getattr(resource, attr, None))
        ]
        if missing:
            logger.warning("Factory produced incompatible MCP client missing %s", ", ".join(missing))
            return None
        client = cast(MCPClientProtocol, resource)
        self._exit_stack.callback(self._safe_close, client)
        return client

    def _refresh_tool_cache(self) -> None:
        cache: "OrderedDict[str, ToolDescriptor]" = OrderedDict()
        for name, entrypoint in self._local_tools.items():
            metadata = self._local_metadata.get(name, {})
            descriptor = ToolDescriptor(
                name=name,
                origin=ToolOrigin.LOCAL,
                entrypoint=entrypoint,
                metadata=metadata,
            )
            cache[name] = descriptor
        remote_bindings: Dict[str, MCPClientProtocol] = {}
        if self._mcp_clients:
            for client in self._mcp_clients:
                for tool in self._iter_client_tools(client):
                    name = tool.get("name") if isinstance(tool, Mapping) else tool
                    if not isinstance(name, str) or not name:
                        continue
                    metadata: Mapping[str, object]
                    if isinstance(tool, Mapping):
                        metadata = {
                            key: value
                            for key, value in tool.items()
                            if key != "name"
                        }
                    else:
                        metadata = {}
                    descriptor = ToolDescriptor(
                        name=name,
                        origin=ToolOrigin.MCP,
                        entrypoint=None,
                        metadata=metadata,
                    )
                    cache[name] = descriptor
                    remote_bindings[name] = client
        self._tool_cache = cache
        if self._mcp_session:
            self._mcp_session.update_tool_cache(remote_bindings)

    def _iter_client_tools(self, client: MCPClientProtocol) -> Iterator[Mapping[str, object] | str]:
        listing: object
        try:
            listing = client.list_tools()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to list tools for MCP client %s", getattr(client, "name", "unknown"), exc_info=True)
            return iter(())
        if asyncio.iscoroutine(listing):
            listing = asyncio.run(listing)
        if isinstance(listing, Mapping):
            listing = [listing]
        if isinstance(listing, Sequence):
            def _iter() -> Iterator[Mapping[str, object] | str]:
                for item in listing:  # type: ignore[assignment]
                    if isinstance(item, (str, Mapping)):
                        yield item  # type: ignore[misc]
            return _iter()
        logger.debug(
            "MCP client %s returned unsupported tool listing %r",
            getattr(client, "name", "unknown"),
            listing,
        )
        return iter(())

    @staticmethod
    def _safe_close(client: MCPClientProtocol) -> None:
        for attr in ("close", "disconnect", "shutdown"):
            hook = getattr(client, attr, None)
            if callable(hook):
                try:
                    hook()
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Error closing MCP client via %s", attr, exc_info=True)
                return


__all__ = [
    "AgentSession",
    "MCPClientProtocol",
    "ToolDescriptor",
    "ToolOrigin",
]
