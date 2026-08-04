from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentTool:
    """Describe one callable capability exposed to the agent router."""

    name: str
    description: str
    handler: Callable[..., Any]


def build_tool_registry(tools: list[AgentTool]) -> dict[str, AgentTool]:
    """Build a name-based registry and reject duplicate tool names."""
    registry = {tool.name: tool for tool in tools}
    if len(registry) != len(tools):
        raise ValueError("Agent tool names must be unique.")
    return registry