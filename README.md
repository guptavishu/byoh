# BYOH — Build Your Own Harness

Thin Python harness around LLM calls. Everything is pluggable: providers, tools, execution modes, system prompts, skills, and evals.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires [Ollama](https://ollama.ai) running locally (default provider).

## Quick start

```bash
# One-shot prompt
byoh "What is 2+2?"

# Interactive chat
byoh

# Streaming (no tools/code, just text)
byoh -s "Tell me a joke"
```

By default, byoh runs in **hybrid mode** — the LLM has both structured tool calls and Python code execution available, and picks which to use per step.

## Execution modes

```bash
# Hybrid (default) — LLM chooses tools or code per step
byoh "Read data.csv and compute the average"

# Tool-only — structured tool calls, no code execution
byoh --tools-only "Read pyproject.toml"

# Code-only — LLM writes Python, byoh runs it in a subprocess
byoh --exec "Calculate the first 20 Fibonacci numbers"
```

Built-in tools: `read_file`, `write_file`. Add more via skills or the Python API.

## Orchestration

Orchestration modes are stackable prompt fragments that modify LLM behavior. Planning is a built-in one; you can add your own.

```bash
# Add planning — LLM breaks the task into numbered steps
byoh --orch planning "Summarize all .py files and write to /tmp/summary.txt"

# Stack multiple orchestrations
byoh --orch planning safety "Refactor the codebase"

# List available modes and skills
byoh --skills
```

## Skills

A skill bundles a system prompt and optionally tools into a reusable package. Skills are auto-loaded from two locations:

- `./byoh_skills.py` — project-local (checked into your repo)
- `~/.byoh/skills.py` — global (available everywhere)

Both files export a `SKILLS` list:

```python
# byoh_skills.py
from byoh import Skill, Tool

SKILLS = [
    # Prompt-only skill — modifies LLM behavior
    Skill(
        name="safety",
        prompt="Never execute destructive commands like rm -rf or drop tables.",
    ),

    # Skill with tools — adds capabilities
    Skill(
        name="web",
        prompt="You can search the web and fetch pages. Use web_search for "
               "discovery, web_fetch for specific URLs.",
        tools=[
            Tool(
                name="web_search",
                description="Search the web",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                fn=my_search_fn,
            ),
        ],
    ),
]
```

Activate skills via `--orch`:

```bash
byoh --orch safety "do something"         # prompt-only skill
byoh --orch web "find the weather in SF"  # skill with tools
byoh --orch web planning "research X"     # combine skills + built-ins
```

Skill tools are **lazy** — they only appear in LLM API calls when the skill is activated. Without `--orch web`, the LLM never sees `web_search`.

## Eval framework

The eval framework compares any combination of execution mode + orchestration across a set of test cases. Ships with 15 built-in cases; add your own in Python or JSON.

```bash
# Run default configs (tool vs code vs hybrid)
python eval_suite.py --verbose

# Compare specific configurations
python eval_suite.py --configs hybrid hybrid+planning

# Compare hybrid with and without a skill
python eval_suite.py --configs hybrid hybrid+safety

# Run only ambiguous cases
python eval_suite.py --configs hybrid hybrid+planning --tags ambiguous

# Load extra cases from JSON
python eval_suite.py --json evals/my_cases.json

# Save results
python eval_suite.py -o results/run1.json
```

### Eval configs

A config is a `mode+orchestration` spec. Use `+` to stack orchestrations:

```
tool                  # tool-only, no orchestration
code                  # code-only
hybrid                # hybrid (default)
hybrid+planning       # hybrid with planning
tool+safety+concise   # tool-only with safety and concise orchestrations
```

### Custom eval cases

In Python:

```python
from byoh.eval import EvalCase, EvalConfig, EvalRunner, Mode

cases = [
    EvalCase(
        name="my_test",
        prompt="Calculate the factorial of 10",
        preferred_mode=Mode.CODE,
        check=lambda r: "3628800" in r.response.content,
        tags=["math"],
    ),
]

runner = EvalRunner(harness, configs=[
    EvalConfig("hybrid"),
    EvalConfig("hybrid", orchestration=["planning"]),
    EvalConfig("code"),
])
results = runner.run_suite(cases)
```

In JSON (loadable via `--json` or `load_cases_from_json()`):

```json
[
    {
        "name": "factorial_test",
        "prompt": "Calculate the factorial of 10",
        "preferred_mode": "code",
        "tags": ["math"]
    }
]
```

### Eval output

The report shows pass/fail per config, round counts, token usage, duration, and which config performed best for each case. Results can be saved as JSON for tracking over time.

## Python API

```python
from byoh import Harness, Skill, Tool, FILE_TOOLS

# Basic usage — defaults to Ollama, hybrid mode
h = Harness()
response = h("What is 2+2?")
print(response.content)

# Streaming
for chunk in h.stream("Tell a joke"):
    print(chunk, end="")

# With tools (hybrid mode by default)
h = Harness(tools=FILE_TOOLS)
response = h.run("Read pyproject.toml and count the lines")

# Force a specific execution mode
response = h.run("Read example.py", tools_only=True)
response = h.run("Calculate fibonacci(20)", exec_code=True)

# Orchestration — stackable prompt modifiers
response = h.run("Summarize the codebase", orchestration=["planning"])
response = h.run("Refactor safely", orchestration=["planning", "safety"])

# Skills — register, then activate via orchestration
web = Skill(name="web", prompt="You can search the web.", tools=[search_tool])
h.add_skill(web)
h.run("Find the weather", orchestration=["web"])

# Register multiple skills at once
h.use(web_skill, safety_skill, concise_skill)

# Callbacks for visibility into tool/code activity
response = h.run(
    "Read example.py",
    on_tool_call=lambda tc: print(f"Calling {tc.name}"),
    on_tool_result=lambda r: print(f"Got: {r.content[:50]}"),
    on_code_exec=lambda code: print(f"Running code..."),
    on_code_result=lambda r: print(f"Output: {r.stdout[:50]}"),
)

# Async
response = await h.acomplete("hello")
response = await h.arun("Summarize files", orchestration=["planning"])
```

## Extending byoh

### Custom providers

Implement the `Provider` protocol — four methods, no inheritance:

```python
from byoh import Harness, Response

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

# Use it
h = Harness(provider=MyProvider())

# Or set as the global default
Harness.default_provider = staticmethod(lambda: MyProvider())
```

### Custom tools

```python
from byoh import Tool, Harness

my_tool = Tool(
    name="shell",
    description="Run a shell command",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    fn=lambda command: subprocess.run(command, shell=True, capture_output=True, text=True).stdout,
)

h = Harness(tools=[my_tool])
```

### Custom system prompts

Override built-in prompts or add new ones:

```python
h = Harness(prompts={
    # Override how the LLM decides between tools and code
    "hybrid": "Always try tools first. Only use code if tools can't do it.",
    # Add a new prompt fragment
    "domain": "You are a data analyst. Prefer pandas for tabular data.",
})

# Activate custom prompts via orchestration
h.run("Analyze sales.csv", orchestration=["domain"])
```

### Custom skills

Package prompts + tools for reuse:

```python
from byoh import Skill, Tool

db_skill = Skill(
    name="database",
    prompt="You can query the PostgreSQL database. Use the query tool for SELECT statements only.",
    tools=[query_tool],
    description="Read-only database access",
)

h = Harness()
h.add_skill(db_skill)
h.run("What tables exist?", orchestration=["database"])
```

Put skills in `byoh_skills.py` for auto-loading by the CLI.

## Defaults

| Setting | Default | Override |
|---------|---------|----------|
| Provider | Ollama (`localhost:11434`) | `Harness(provider=...)` or `Harness.default_provider = ...` |
| Model | `qwen2.5-coder:32b` | `-m model_name` or `Harness(model=...)` |
| Execution mode | Hybrid (tools + code) | `--tools-only`, `--exec`, or `tools_only=True`, `exec_code=True` |
| Tools | `read_file`, `write_file` | `Harness(tools=[...])` or via skills |
| Orchestration | None | `--orch name1 name2` or `orchestration=[...]` |
| Skills | Auto-loaded from `./byoh_skills.py` and `~/.byoh/skills.py` | `h.add_skill(...)` or `h.use(...)` |
| System prompt | None | `--system "..."` or `Harness(system=...)` |
| Temperature | 0.0 | `Harness(temperature=...)` |
| Max tokens | 4096 | `Harness(max_tokens=...)` |

## Logs

Every LLM call, tool execution, and code execution is logged to `~/.byoh/byoh.log` as JSON lines.

```bash
tail -f ~/.byoh/byoh.log | jq .
```

## Roadmap

These are things that don't exist yet but could be built on top of the current architecture. Contributions welcome.

- **Lifecycle hooks** — before/after each LLM call, on loop iteration, on max rounds reached. Would enable middleware patterns like retries, rate limiting, token budgets, and guardrails.
- **Pluggable execution sandbox** — `execute_code()` currently runs Python in a local subprocess. An interface for Docker, WASM, or remote execution would improve safety.
- **Memory** — persistent conversation history across sessions. Could be a skill that loads/saves context from `~/.byoh/memory/`.
- **More built-in tools** — shell commands, web fetch, grep/search. Could ship as built-in skills rather than hardcoded tools.
- **Context engineering** — retrieval injection, context window budgets, automatic summarization of long conversations.
- **Hybrid prompt tuning** — eval data shows the LLM sometimes embeds tool response XML into code blocks. The hybrid system prompt could be improved to prevent this.

## CLI reference

```
byoh [prompt] [-m MODEL] [-s] [--tools-only] [--exec] [--orch MODE ...] [--system SYSTEM] [--skills]

positional:
  prompt              Prompt text (omit for interactive mode)

options:
  -m, --model         Model override
  -s, --stream        Stream output (no tools/code)
  --tools-only        Tool-only mode (no code execution)
  --exec              Code-only mode (no tools)
  --orch MODE ...     Orchestration modes to stack (e.g. planning, safety)
  --system            Custom system prompt
  --skills            List available skills and orchestration modes
```

### Eval CLI

```
python eval_suite.py [--configs SPEC ...] [--tags TAG ...] [--cases NAME ...] [--json PATH] [-m MODEL] [-v] [-o PATH] [--list]

options:
  --configs SPEC ...  Configs to compare (e.g. hybrid hybrid+planning tool code)
  --tags TAG ...      Filter cases by tag
  --cases NAME ...    Run specific cases by name
  --json PATH         Load additional cases from a JSON file
  -m, --model         Model override
  -v, --verbose       Show progress
  -o, --output        Save results JSON to this path
  --list              List available cases and exit
```
