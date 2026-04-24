# BYOH — Build Your Own Harness

## What This Is
Thinnest possible Python harness around LLM calls. Pluggable provider architecture with hybrid tool/code execution — the LLM decides whether to use structured tool calls or Python code per step.

## Current State
v0.5 — 9 source files. Sync + async, streaming + one-shot, Ollama provider (bring your own for others), file tools, hybrid/tool-only/code-only modes, multi-step planning, code execution, eval framework, structured logging, CLI with interactive chat mode.

## Structure

```
byoh/
├── pyproject.toml          # Package config, depends on httpx>=0.27.0
├── eval_suite.py           # Eval suite — 15 cases comparing tool/code/hybrid modes
├── README.md               # User-facing documentation
└── byoh/
    ├── __init__.py          # Re-exports: Harness, Message, Provider, Response, OllamaProvider
    ├── types.py             # Message, Response, Tool, ToolCall, ToolResult + Provider protocol
    ├── ollama.py            # OllamaProvider — talks to local Ollama via REST API
    ├── tools.py             # ToolRegistry + built-in read_file/write_file tools
    ├── executor.py          # Code execution engine — runs LLM-emitted Python in subprocess
    ├── eval.py              # Eval framework — EvalCase, EvalRunner, grading, reporting
    ├── logging.py           # Structured JSON logging to ~/.byoh/byoh.log
    ├── cli.py               # CLI entry point — hybrid by default, tool-only/code-only overrides
    └── core.py              # Harness class — main entry point, hybrid/tool/code loops
```

## Data Flow

```
"prompt string" → Harness._to_messages() → [Message] → Provider.complete() → Response

Hybrid loop (run(), default):  LLM → tool_calls OR code blocks → execute → feed back → ... → final text
Tool loop (run(tools_only=True)):  LLM → tool_calls → execute → feed results → LLM → ... → final text
Code loop (run(exec_code=True)):   LLM → code blocks → subprocess → feed stdout → LLM → ... → final text
```

## Key Design Decisions
- **Provider is a typing.Protocol** — structural subtyping, no inheritance needed. Implement 4 methods: complete(), stream(), acomplete(), astream().
- **Bring your own provider** — only Ollama ships built-in. Plug in any LLM by implementing the Provider protocol.
- **Hybrid mode is the default** — the LLM has both tools and code execution available and picks per step. Override with `tools_only=True` or `exec_code=True`.
- **Harness is callable** — `h("prompt")` works as shorthand for `h.complete("prompt")`.
- **System prompt lives on Harness**, not in messages — keeps message list clean for multi-turn.
- **Response is flat** — no provider-specific fields. Every provider maps to the same shape (content, model, stop_reason, usage, tool_calls).
- **Planning and code execution are system prompt injections** — no new classes, just `plan=True` on `run()`.
- **Default model**: qwen2.5-coder:32b (Ollama).

## Usage

```python
from byoh import Harness
h = Harness()                          # defaults to Ollama
response = h("What is 2+2?")          # sync one-shot
for chunk in h.stream("joke"):        # sync streaming
    print(chunk, end="")
response = await h.acomplete("hi")    # async one-shot
async for c in h.astream("hi"):       # async streaming
    print(c, end="")
```

## Requires
- Python >=3.11
- Ollama running locally (if using default provider)
- `pip install -e .` to install

## Completed Milestones
- **v0.1** — Core harness, Ollama provider, CLI
- **v0.2** — File tools (read_file, write_file), tool-use loop
- **v0.3** — Multi-step planning via `plan=True`
- **v0.4** — Code execution (LLMVM-style) via `exec_code=True`
- **v0.5** — Hybrid mode (default), eval framework, removed Claude dependency

## Potential Next Steps

### Extensibility
- **Pluggable system prompts** — HYBRID_PROMPT, CODE_EXEC_PROMPT, PLANNING_PROMPT are hardcoded in core.py. Make them overridable on Harness so users can tune how the LLM picks between tools and code without editing source.
- **Pluggable execution sandbox** — execute_code() always runs Python in a local subprocess. Interface for Docker, WASM, or remote execution.
- **Lifecycle hooks** — "before each LLM call", "on loop iteration", "on max rounds reached". Current callbacks (on_tool_call etc.) are per-action, not per-loop-step.
- **Skills** — reusable prompt+tool bundles for specific domains. A skill packages a system prompt fragment, a set of tools, and optionally instructions for when to activate. The harness would compose skills into the final system prompt and tool set.

### Features
- Memory — persistent conversation history across sessions
- More tools — shell commands, web fetch, search/grep
- Context engineering — retrieval injection, context budgets
- Better sandboxing — Docker containers for code execution
- Tune hybrid system prompt — fix the tool-response-in-code-block issue

## User Preferences
- Add comments to code — user explicitly requested this
- Keep it as thin as possible — no unnecessary abstractions
- Python was chosen for ecosystem breadth and prototyping speed
