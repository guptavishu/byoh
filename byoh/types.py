"""Core types for the harness.

What lives here:
  - Message: what goes into a provider (text or tool results)
  - Response: what comes back (text and/or tool calls)
  - Tool, ToolCall, ToolResult: tool-use plumbing
  - Provider: the interface any LLM backend must implement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Message:
    """A single turn in a conversation.

    role is "user", "assistant", or "tool" — system prompts are passed
    separately so the message list stays clean for multi-turn.

    For tool results, role="tool" and tool_call_id + content are set.
    For assistant messages with tool calls, tool_calls is populated.
    """

    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the LLM can invoke.

    parameters uses JSON Schema format — the standard that most LLM
    APIs expect, so no per-provider translation needed.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., str]


@dataclass(frozen=True, slots=True)
class Response:
    """What the provider hands back after a completion.

    Intentionally flat — no provider-specific fields.
    tool_calls is non-empty when the LLM wants to invoke tools.
    """

    content: str
    model: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)  # {"input": N, "output": N}
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Skill:
    """A reusable bundle of prompt + tools for a specific domain.

    A skill packages everything the LLM needs for a capability:
    a system prompt fragment and optionally a set of tools. When
    added to a Harness, the prompt registers under the skill's name
    and the tools are added to the registry. Activate via orchestration.

        web = Skill(
            name="web",
            prompt="You can fetch web pages and search the internet.",
            tools=[search_tool, fetch_tool],
        )
        h = Harness()
        h.add_skill(web)
        h.run("Find the weather", orchestration=["web"])
    """

    name: str
    prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    description: str = ""


@runtime_checkable
class Provider(Protocol):
    """The contract any LLM backend must satisfy.

    Uses typing.Protocol (structural subtyping) so providers don't need
    to inherit from anything — just implement these four methods.

    Two pairs: sync + async, each with a one-shot and streaming variant.
    tools is optional — when provided, the LLM can request tool calls.
    """

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Synchronous, non-streaming completion."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Iterator[str]:
        """Synchronous streaming — yields text chunks."""
        ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Async non-streaming completion."""
        ...

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming — yields text chunks."""
        ...
