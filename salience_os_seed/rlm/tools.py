"""Tool definitions for the recursive language model orchestration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from .types import RunContext, ToolSpec


def build_toolkit(store) -> Sequence[ToolSpec]:
    """Construct the default tool suite backed by the provided store."""

    return (
        ReadTool(store),
        ScanTool(store),
        SummarizeTool(store),
        WriteTool(store),
        SpawnTool(),
    )


class BaseTool(ToolSpec):
    def __init__(self, name: str, description: str, schema: Mapping[str, Any]) -> None:
        super().__init__(name=name, description=description, schema=schema, handler=self._call)

    def descriptor(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
        }

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class ReadTool(BaseTool):
    def __init__(self, store) -> None:
        super().__init__(
            name="read",
            description="Load a slice of an artifact by id/path.",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        )
        self._store = store

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> str:
        identifier = str(args["id"])
        start = args.get("start")
        end = args.get("end")
        return self._store.read(identifier, start=start, end=end)


class ScanTool(BaseTool):
    def __init__(self, store) -> None:
        super().__init__(
            name="scan",
            description="Search the workspace and return top hits.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "topK": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        self._store = store

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> Sequence[Mapping[str, Any]]:
        query = str(args["query"])
        top_k = int(args.get("topK", 6))
        top_k = max(1, min(top_k, 20))
        return self._store.scan(query, top_k=top_k)


class SummarizeTool(BaseTool):
    def __init__(self, store) -> None:
        super().__init__(
            name="summarize",
            description="Produce a lossy summary of an artifact within budget.",
            schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "targetLen": {"type": "integer", "minimum": 50, "maximum": 2000},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        )
        self._store = store

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> str:
        identifier = str(args["id"])
        target_len = int(args.get("targetLen", 400))
        return self._store.summarize(identifier, target_len=target_len)


class WriteTool(BaseTool):
    def __init__(self, store) -> None:
        super().__init__(
            name="write",
            description="Persist scratch notes and findings into the workspace scratch area.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "data": {},
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        )
        self._store = store

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
        path = args.get("path")
        data = args.get("data", "")
        return self._store.write(path, data)


class SpawnTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="spawn",
            description="Propose child tasks with scoped identifiers and local budgets.",
            schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "scope": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "integer", "minimum": 50},
                    "maxDepth": {"type": "integer", "minimum": 0},
                    "children": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "scope": {"type": "array", "items": {"type": "string"}},
                                "budget": {"type": "integer"},
                                "salience": {"type": "number"},
                            },
                            "required": ["task"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    def _call(self, args: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
        task = str(args["task"])
        scope = ctx.store.clamp_scope(args.get("scope"))
        budget = int(args.get("budget", ctx.policy.per_call_cap))
        children = []
        for child in args.get("children", []):
            children.append(
                {
                    "task": str(child.get("task", task)),
                    "scope": ctx.store.clamp_scope(child.get("scope")),
                    "budget": int(child.get("budget", budget // max(1, len(args.get("children", [])) or 1))),
                    "salience": float(child.get("salience", 0.0)),
                }
            )
        return {
            "task": task,
            "scope": scope,
            "budget": budget,
            "children": children,
            "maxDepth": int(args.get("maxDepth", 0)),
        }


def tool_descriptors(toolkit: Iterable[ToolSpec]) -> Sequence[Dict[str, Any]]:
    return [tool.descriptor() for tool in toolkit]
