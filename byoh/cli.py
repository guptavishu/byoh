"""Minimal CLI for the byoh.

Usage:
    byoh "What is 2+2?"                           # hybrid mode (default)
    byoh -s "Tell me a joke"                       # streaming (no tools/code)
    byoh --orch planning "Summarize all .py files" # with planning orchestration
    byoh --orch planning safety "do something"     # stack multiple orchestrations
    byoh --tools-only "Read example.py"            # tool-only mode
    byoh --exec "List files in this dir"           # code-only mode
    byoh                                           # interactive hybrid mode

Skills are auto-loaded from:
    ./byoh_skills.py   — project-local skills
    ~/.byoh/skills.py  — global skills

Both files should export a SKILLS list:
    from byoh import Skill
    SKILLS = [Skill(name="safety", prompt="Never delete files.")]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .core import Harness
from .executor import ExecResult
from .tools import FILE_TOOLS
from .types import Message, Skill, ToolCall, ToolResult


def _load_skills_from(path: Path) -> list[Skill]:
    """Load skills from a Python file that exports a SKILLS list."""
    if not path.is_file():
        return []
    spec = importlib.util.spec_from_file_location(f"byoh_skills_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
        return []
    skills = getattr(mod, "SKILLS", [])
    return [s for s in skills if isinstance(s, Skill)]


def _load_skills(harness: Harness) -> list[str]:
    """Auto-discover and register skills. Returns list of loaded skill names."""
    loaded = []
    # Global skills first, then project-local (local overrides global)
    for path in [Path.home() / ".byoh" / "skills.py", Path("byoh_skills.py")]:
        for skill in _load_skills_from(path):
            harness.add_skill(skill)
            loaded.append(skill.name)
    return loaded


def _on_tool_call(tc: ToolCall) -> None:
    """Print when the LLM calls a tool."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
    print(f"  → {tc.name}({args_str})")


def _on_tool_result(result: ToolResult) -> None:
    """Print the tool result (truncated for readability)."""
    preview = result.content[:200]
    if len(result.content) > 200:
        preview += "..."
    status = "ERROR" if result.is_error else "OK"
    print(f"  ← [{status}] {preview}")


def _on_code_exec(code: str) -> None:
    """Print when a code block is about to be executed."""
    lines = code.strip().splitlines()
    preview = "\n".join(lines[:3])
    if len(lines) > 3:
        preview += f"\n    ... ({len(lines)} lines)"
    print(f"  ▶ executing:")
    for line in preview.splitlines():
        print(f"    {line}")


def _on_code_result(result: ExecResult) -> None:
    """Print the result of code execution."""
    status = "OK" if result.ok else f"FAILED (rc={result.returncode})"
    output = result.output[:300]
    if len(result.output) > 300:
        output += "..."
    print(f"  ◀ [{status}]")
    if output.strip():
        for line in output.splitlines()[:10]:
            print(f"    {line}")


def _interactive(byoh: Harness, *, tools_only: bool, orchestration: list[str], exec_code: bool):
    """Interactive chat loop with conversation history."""
    history: list[Message] = []
    if exec_code:
        mode = "exec"
    elif tools_only:
        mode = "tools"
    else:
        mode = "hybrid"
    if orchestration:
        mode += "+" + "+".join(orchestration)
    print(f"byoh {mode} (ctrl-c to quit)")
    print()
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input.strip():
            continue
        history.append(Message(role="user", content=user_input))

        response = byoh.run(
            history,
            orchestration=orchestration or None,
            tools_only=tools_only,
            exec_code=exec_code,
            on_tool_call=_on_tool_call,
            on_tool_result=_on_tool_result,
            on_code_exec=_on_code_exec,
            on_code_result=_on_code_result,
        )

        print(response.content)
        history.append(Message(role="assistant", content=response.content))
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="byoh",
        description="Thin CLI for LLM calls — hybrid mode by default",
    )
    parser.add_argument("prompt", nargs="?", help="prompt text (omit for interactive mode)")
    parser.add_argument("-m", "--model", default=None, help="model override")
    parser.add_argument("-s", "--stream", action="store_true", help="stream output (no tools/code)")
    parser.add_argument("--tools-only", action="store_true",
                        help="tool-only mode — no code execution")
    parser.add_argument("--exec", dest="exec_code", action="store_true",
                        help="code-only mode — no tools")
    parser.add_argument("--orch", nargs="*", default=[], metavar="MODE",
                        help="orchestration modes to stack (e.g. planning, safety)")
    parser.add_argument("--system", default=None, help="system prompt")
    parser.add_argument("--skills", action="store_true",
                        help="list available skills and orchestration modes, then exit")
    args = parser.parse_args()

    tools = None if args.exec_code else FILE_TOOLS
    h = Harness(system=args.system, model=args.model, tools=tools)

    # Auto-load skills from byoh_skills.py and ~/.byoh/skills.py
    loaded = _load_skills(h)

    if args.skills:
        from byoh.core import BUILTIN_PROMPTS, DEFAULT_ORCHESTRATION
        print("Available orchestration modes and skills:\n")
        print("  Built-in:")
        for key in BUILTIN_PROMPTS:
            default_marker = " *" if key in DEFAULT_ORCHESTRATION else ""
            preview = h.prompts.get(key, "")[:60].replace("\n", " ")
            print(f"    {key:<20} {preview}...{default_marker}")
        if loaded:
            print("\n  From skills:")
            for name in loaded:
                preview = h.prompts.get(name, "")[:60].replace("\n", " ")
                print(f"    {name:<20} {preview}...")
        else:
            print("\n  No skills loaded.")
            print("  Add skills to ./byoh_skills.py or ~/.byoh/skills.py")
        if DEFAULT_ORCHESTRATION:
            print(f"\n  * = active by default ({', '.join(DEFAULT_ORCHESTRATION)})")
        return

    if args.prompt is None:
        _interactive(h, tools_only=args.tools_only, orchestration=args.orch,
                     exec_code=args.exec_code)
        return

    if args.stream:
        for chunk in h.stream(args.prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        response = h.run(
            args.prompt,
            orchestration=args.orch or None,
            tools_only=args.tools_only,
            exec_code=args.exec_code,
            on_tool_call=_on_tool_call,
            on_tool_result=_on_tool_result,
            on_code_exec=_on_code_exec,
            on_code_result=_on_code_result,
        )
        print(response.content)


if __name__ == "__main__":
    main()
