"""byoh — Build Your Own Harness. Thinnest possible wrapper around LLM calls.

Usage:
    from byoh import Harness

    h = Harness()                         # defaults to Ollama
    response = h("What is 2+2?")          # sync one-shot
    print(response.content)

    for chunk in h.stream("Tell a joke"): # sync streaming
        print(chunk, end="")

Bring your own provider — implement the Provider protocol:

    from byoh import Harness, Provider, Message, Response

    class MyProvider:
        def complete(self, messages, *, system=None, model=None,
                     temperature=0.0, max_tokens=4096, tools=None) -> Response:
            ...  # call your LLM, return a Response

        def stream(self, messages, **kwargs):
            ...  # yield str chunks

        async def acomplete(self, messages, **kwargs) -> Response:
            ...

        async def astream(self, messages, **kwargs):
            ...  # async yield str chunks

    h = Harness(provider=MyProvider())
"""

from .types import Message, Provider, Response, Skill, Tool, ToolCall, ToolResult
from .core import Harness
from .tools import ToolRegistry, FILE_TOOLS


def __getattr__(name: str):
    # Lazy import — OllamaProvider is only loaded when accessed
    if name == "OllamaProvider":
        from .ollama import OllamaProvider
        return OllamaProvider
    raise AttributeError(f"module 'byoh' has no attribute {name!r}")


__all__ = [
    "Harness", "Message", "Provider", "Response", "Skill",
    "Tool", "ToolCall", "ToolResult",
    "OllamaProvider", "ToolRegistry", "FILE_TOOLS",
]
