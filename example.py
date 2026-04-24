"""Quick examples showing byoh usage."""

import asyncio

from byoh import Harness, Message, OllamaProvider, FILE_TOOLS


def basic_examples():
    """Sync one-shot, streaming, and multi-turn with Ollama."""
    h = Harness(
        provider=OllamaProvider(),
        system="You are a helpful assistant. Be concise.",
    )

    # One-shot
    response = h("What is 2+2?")
    print(f"Response: {response.content}")
    print(f"Model: {response.model}, Tokens: {response.usage}")

    # Streaming
    print("\nStreaming: ", end="")
    for chunk in h.stream("Tell me a one-line joke"):
        print(chunk, end="", flush=True)
    print()

    # Multi-turn conversation
    conversation = [
        Message(role="user", content="My name is Alice."),
        Message(role="assistant", content="Hello Alice! How can I help you?"),
        Message(role="user", content="What's my name?"),
    ]
    response = h(conversation)
    print(f"\nMulti-turn: {response.content}")


def tool_examples():
    """Using file tools — the LLM reads/writes files."""
    h = Harness(
        provider=OllamaProvider(),
        tools=FILE_TOOLS,
    )

    # Tool-use loop: LLM decides to call read_file, byoh executes it
    response = h.run("Read pyproject.toml and tell me the project version")
    print(f"\nTool result: {response.content}")

    # With planning: LLM breaks task into steps
    response = h.run(
        "Read types.py and core.py, summarize both",
        plan=True,
    )
    print(f"\nPlanned result: {response.content}")


def exec_example():
    """Code execution mode — LLM writes Python, byoh runs it."""
    h = Harness(provider=OllamaProvider())

    response = h.run(
        "Count the total lines of Python code in byoh/",
        exec_code=True,
    )
    print(f"\nExec result: {response.content}")


async def async_examples():
    """Async variants of the above."""
    h = Harness(provider=OllamaProvider())

    response = await h.acomplete("What is the capital of France?")
    print(f"\nAsync: {response.content}")

    print("Async streaming: ", end="")
    async for chunk in h.astream("Count to 5"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    basic_examples()
    tool_examples()
    exec_example()
    asyncio.run(async_examples())
