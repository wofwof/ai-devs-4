from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolResult:
    """Result returned by a tool handler.

    data: ready-to-use content (goes straight to agent), or a prompt for a sub-LLM call.
    llm_schema: if set, the main loop makes an LLM call with `data` as prompt
                and this as response_format, then feeds the LLM output to the agent.
    stop: if True, the main loop should stop after this tool call.
    """

    data: str
    llm_schema: dict | None = None
    stop: bool = False

    @property
    def needs_llm(self) -> bool:
        return self.llm_schema is not None


@dataclass
class _Tool:
    """Internal representation of a registered tool."""

    schema: dict
    handler: Callable[..., ToolResult]


class ToolRegistry:
    """Central registry for OpenAI function-calling tools and their handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, _Tool] = {}

    def register(
        self,
        schema: dict,
        handler: Callable[..., ToolResult],
    ) -> None:
        """Register a tool with its OpenAI schema and handler function.

        The tool name is extracted from the schema automatically.
        """
        name = schema["function"]["name"]
        self._tools[name] = _Tool(schema=schema, handler=handler)

    def get_tools(self, *names: str) -> list[dict]:
        """Return OpenAI tool schemas for the given tool names.

        If no names are provided, returns all registered tool schemas.
        """
        if not names:
            return [t.schema for t in self._tools.values()]
        missing = set(names) - self._tools.keys()
        if missing:
            raise KeyError(f"Unknown tools: {missing}")
        return [self._tools[n].schema for n in names]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool handler by name with the given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(data=f"Error: unknown tool '{name}'")
        return tool.handler(**arguments)

    @property
    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())


# Global registry instance
registry = ToolRegistry()
