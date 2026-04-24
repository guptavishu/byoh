"""Ollama provider implementation.

Talks to a local Ollama instance via its REST API (default: localhost:11434).
Uses httpx for HTTP communication with the Ollama REST API.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator

import httpx

from .types import Message, Response, Tool, ToolCall

# Default model — change to whatever you have pulled locally
DEFAULT_MODEL = "qwen2.5-coder:32b"
DEFAULT_BASE_URL = "http://localhost:11434"


def _to_api_messages(
    messages: list[Message], *, system: str | None
) -> list[dict]:
    """Convert our Messages to the Ollama chat format.

    Handles regular messages, assistant messages with tool calls,
    and tool result messages.
    """
    api_msgs: list[dict] = []
    if system:
        api_msgs.append({"role": "system", "content": system})

    for m in messages:
        if m.role == "tool":
            # Tool result — Ollama expects role="tool" with content
            api_msgs.append({
                "role": "tool",
                "content": m.content,
            })
        elif m.role == "assistant" and m.tool_calls:
            # Assistant message that requested tool calls
            api_msgs.append({
                "role": "assistant",
                "content": m.content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            api_msgs.append({"role": m.role, "content": m.content})

    return api_msgs


def _tools_to_api(tools: list[Tool]) -> list[dict]:
    """Convert our Tool objects to Ollama's tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_tool_calls(raw_calls: list[dict]) -> list[ToolCall]:
    """Parse Ollama's tool_calls into our ToolCall type."""
    result = []
    for i, tc in enumerate(raw_calls):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        # Ollama sometimes returns args as a JSON string
        if isinstance(args, str):
            args = json.loads(args)
        result.append(ToolCall(
            id=f"call_{i}",
            name=func.get("name", ""),
            arguments=args,
        ))
    return result


def _try_parse_content_as_tool_call(content: str, tools: list[Tool] | None) -> tuple[list[ToolCall], str]:
    """Some models return tool calls as JSON in content instead of using
    the structured tool_calls field. Detect and parse these.

    Handles both pure-JSON content and JSON embedded in text.
    Returns (tool_calls, remaining_text) — remaining_text is any
    non-tool-call text the model also produced.
    """
    if not tools or not content.strip():
        return [], content

    tool_names = {t.name for t in tools}

    # First try: entire content is JSON
    try:
        data = json.loads(content.strip())
        if isinstance(data, dict) and data.get("name") in tool_names:
            return [ToolCall(
                id="call_0",
                name=data["name"],
                arguments=data.get("arguments", data.get("parameters", {})),
            )], ""
        if isinstance(data, list):
            calls = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and item.get("name") in tool_names:
                    calls.append(ToolCall(
                        id=f"call_{i}",
                        name=item["name"],
                        arguments=item.get("arguments", item.get("parameters", {})),
                    ))
            if calls:
                return calls, ""
    except json.JSONDecodeError:
        pass

    # Second try: extract JSON objects embedded in text by finding
    # balanced braces that parse as valid tool calls
    calls = []
    remaining_parts = []
    i = 0
    text = content

    while i < len(text):
        if text[i] == '{':
            # Find the matching closing brace
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j + 1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict) and data.get("name") in tool_names:
                                calls.append(ToolCall(
                                    id=f"call_{len(calls)}",
                                    name=data["name"],
                                    arguments=data.get("arguments", data.get("parameters", {})),
                                ))
                                i = j + 1
                                break
                        except json.JSONDecodeError:
                            pass
                        # Not valid JSON or not a tool call — treat as text
                        remaining_parts.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                # No matching brace found
                remaining_parts.append(text[i:])
                break
        else:
            # Accumulate non-JSON text
            next_brace = text.find('{', i)
            if next_brace == -1:
                remaining_parts.append(text[i:])
                break
            remaining_parts.append(text[i:next_brace])
            i = next_brace

    remaining = "".join(remaining_parts).strip()
    return calls, remaining


def _build_payload(
    messages: list[Message],
    *,
    system: str | None,
    model: str | None,
    default_model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    tools: list[Tool] | None = None,
) -> dict:
    """Build the JSON payload for the Ollama /api/chat endpoint."""
    payload: dict[str, Any] = {
        "model": model or default_model,
        "messages": _to_api_messages(messages, system=system),
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if tools:
        payload["tools"] = _tools_to_api(tools)
    return payload


class OllamaProvider:
    """Ollama provider — talks to a locally running Ollama server.

    No API key needed. Just have Ollama running with a model pulled.
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.default_model = default_model
        self.base_url = base_url

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
        payload = _build_payload(
            messages,
            system=system,
            model=model,
            default_model=self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            tools=tools,
        )
        # Ollama can be slow for large models — generous timeout
        resp = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=300.0
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse tool calls — check structured field first, then content fallback
        tool_calls = []
        if raw_calls := data.get("message", {}).get("tool_calls"):
            tool_calls = _parse_tool_calls(raw_calls)

        content = data["message"].get("content", "")

        # Fallback: some models put tool calls as JSON in content
        if not tool_calls and tools:
            tool_calls, remaining = _try_parse_content_as_tool_call(content, tools)
            if tool_calls:
                content = remaining

        return Response(
            content=content,
            model=data["model"],
            stop_reason=data.get("done_reason"),
            usage={
                "input": data.get("prompt_eval_count", 0),
                "output": data.get("eval_count", 0),
            },
            tool_calls=tool_calls,
        )

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
        payload = _build_payload(
            messages,
            system=system,
            model=model,
            default_model=self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            tools=tools,
        )
        # Stream NDJSON lines from Ollama
        with httpx.stream(
            "POST", f"{self.base_url}/api/chat", json=payload, timeout=300.0
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if content := chunk.get("message", {}).get("content", ""):
                    yield content

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
        payload = _build_payload(
            messages,
            system=system,
            model=model,
            default_model=self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            tools=tools,
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=300.0
            )
            resp.raise_for_status()
            data = resp.json()

        tool_calls = []
        if raw_calls := data.get("message", {}).get("tool_calls"):
            tool_calls = _parse_tool_calls(raw_calls)

        content = data["message"].get("content", "")

        # Fallback: some models put tool calls as JSON in content
        if not tool_calls and tools:
            tool_calls, remaining = _try_parse_content_as_tool_call(content, tools)
            if tool_calls:
                content = remaining

        return Response(
            content=content,
            model=data["model"],
            stop_reason=data.get("done_reason"),
            usage={
                "input": data.get("prompt_eval_count", 0),
                "output": data.get("eval_count", 0),
            },
            tool_calls=tool_calls,
        )

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
        payload = _build_payload(
            messages,
            system=system,
            model=model,
            default_model=self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            tools=tools,
        )
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=300.0
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content", ""):
                        yield content
