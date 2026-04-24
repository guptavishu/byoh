"""The Harness — a thin convenience layer over a Provider.

This is the main entry point. It lets you do:

    h = Harness()
    response = h("What is 2+2?")          # sync one-shot
    for chunk in h.stream("Tell a joke"):  # sync streaming
        print(chunk, end="")

The Harness adds two things on top of a raw Provider:
  1. Accepts a plain string (auto-wraps in a Message)
  2. Carries defaults (system prompt, model) so call sites stay clean

The run() method adds a tool-use loop: call LLM → execute tool calls →
feed results back → repeat until the LLM stops calling tools.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from .types import Message, Provider, Response, Skill, Tool
from .tools import ToolRegistry
from .executor import extract_code_blocks, execute_code, CODE_EXEC_PROMPT, ExecResult
from . import logging as hlog
from .logging import Trajectory


# Max iterations to prevent infinite tool-use loops
MAX_TOOL_ROUNDS = 20

# ── Built-in prompt fragments ────────────────────────────────────────
# These are the defaults for Harness.prompts. Users can override any
# of them or add new ones. The _build_system_prompt method assembles
# the final system prompt from these fragments based on the mode.

BUILTIN_PROMPTS: dict[str, str] = {
    "hybrid": """You have two ways to take action:

1. **Tools** — use tool calls for simple, discrete operations like reading or writing
   a single file. Tools are structured and reliable.
2. **Code** — write Python in fenced ```python blocks for computation, data
   transformation, multi-step logic, or anything that benefits from loops,
   variables, or library imports. Code runs in a subprocess and stdout is
   captured.

Pick the right one for each step. You can mix them across steps — use a tool to
read a file, then code to process the data, then a tool to write the result.
Within a single response, use EITHER a tool call OR a code block, not both.""",

    "code_exec": CODE_EXEC_PROMPT,

    "planning": """You are a methodical assistant that breaks tasks into steps.

When given a task:
1. First, think through what steps are needed and output a numbered plan.
2. Execute each step one at a time using the available tools.
3. After each step, briefly state what you did and what's next.
4. If a step fails or reveals new information, revise the remaining plan.
5. When all steps are complete, summarize what was accomplished.

Be concise in your planning — short numbered lists, not paragraphs.""",

    "repair": """Before giving your final answer, critically review your work:
1. Re-read the original request — did you actually do what was asked?
2. Check for errors: wrong file paths, broken code, incorrect calculations,
   missing steps, or incomplete output.
3. If you find a problem, fix it immediately — re-run tools or code as needed.
   Do NOT just describe the problem; actually correct it.
4. Only give your final answer once you are confident it is correct.
5. If you made corrections, briefly note what you fixed.""",
}

# Default orchestration applied to every run() call unless overridden
DEFAULT_ORCHESTRATION = ["repair"]


def _default_provider() -> Provider:
    """Create the default provider (Ollama). Override Harness.default_provider to change."""
    from .ollama import OllamaProvider
    return OllamaProvider()


class Harness:
    """Thin wrapper around any Provider.

    Pass any object that implements the Provider protocol (complete,
    stream, acomplete, astream). When no provider is given, calls
    default_provider to create one (Ollama by default).

    System prompts are composable via the `prompts` dict. Built-in keys:
      "hybrid", "code_exec", "planning"
    Add your own and they'll be included when their key is in the
    active prompt list for a given mode:

        h = Harness()
        h.prompts["safety"] = "Never execute destructive commands."
        h.prompt_plan = ["hybrid", "code_exec", "safety"]  # custom assembly

    To change the default provider globally:
        Harness.default_provider = staticmethod(lambda: MyProvider())
    """

    # Override this to change what provider is used when none is passed
    default_provider: callable = staticmethod(_default_provider)

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        prompts: dict[str, str] | None = None,
    ):
        if provider is None:
            provider = self.default_provider()

        self.provider = provider
        # Defaults that apply to every call unless overridden
        self.system = system
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Tool registry — populated when tools are provided
        self.registry = ToolRegistry(tools) if tools else None
        # Composable prompt fragments — start with built-ins, merge user overrides
        self.prompts: dict[str, str] = dict(BUILTIN_PROMPTS)
        if prompts:
            self.prompts.update(prompts)
        # Registered skills — tools are only activated when orchestration includes them
        self.skills: dict[str, Skill] = {}
        # Default orchestration — applied to every run() unless overridden
        self.default_orchestration: list[str] = list(DEFAULT_ORCHESTRATION)

    def add_skill(self, skill: Skill) -> None:
        """Register a skill.

        The prompt is stored under the skill's name in self.prompts.
        The skill's tools are held back — they only appear in API calls
        when the skill's name is included in the orchestration list.
        """
        self.skills[skill.name] = skill
        if skill.prompt:
            self.prompts[skill.name] = skill.prompt

    def use(self, *skills: Skill) -> None:
        """Register multiple skills at once.

            h = Harness()
            h.use(web_skill, safety_skill)
        """
        for skill in skills:
            self.add_skill(skill)

    def _build_system_prompt(self, keys: list[str], **kwargs) -> str:
        """Assemble a system prompt from named fragments.

        Looks up each key in self.prompts, joins with double newlines,
        then appends the user's base system prompt (from self.system
        or kwargs override) at the end.
        """
        parts = [self.prompts[k] for k in keys if k in self.prompts]
        base = kwargs.get("system", self.system) or ""
        if base:
            parts.append(base)
        return "\n\n".join(parts)

    def _to_messages(self, prompt: str | list[Message]) -> list[Message]:
        """Normalize input: accept a bare string or a message list."""
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return prompt

    def _merge_kwargs(self, *, orchestration: list[str] | None = None, **overrides) -> dict:
        """Merge per-call overrides with instance defaults.

        When orchestration names match registered skills, those skills'
        tools are included in the tool list for this call only.
        """
        kwargs = {
            "system": overrides.get("system", self.system),
            "model": overrides.get("model", self.model),
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        # Base tools from the registry
        base_tools = list(self.registry.tools) if self.registry else []
        # Add tools from active skills
        for name in (orchestration or []):
            skill = self.skills.get(name)
            if skill and skill.tools:
                base_tools.extend(skill.tools)
        if base_tools:
            kwargs["tools"] = base_tools
        return kwargs

    # ── Sync ────────────────────────────────────────────────────────

    def __call__(
        self,
        prompt: str | list[Message],
        **kwargs,
    ) -> Response:
        """Callable shorthand: h("prompt") → Response."""
        return self.complete(prompt, **kwargs)

    def complete(
        self,
        prompt: str | list[Message],
        **kwargs,
    ) -> Response:
        """Synchronous one-shot completion."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(**kwargs)
        hlog.log_llm_request(
            messages,
            model=merged.get("model"),
            system=merged.get("system"),
            temperature=merged.get("temperature", 0.0),
            max_tokens=merged.get("max_tokens", 4096),
            tools=merged.get("tools"),
        )
        response = self.provider.complete(messages, **merged)
        hlog.log_llm_response(response)
        return response

    def stream(
        self,
        prompt: str | list[Message],
        **kwargs,
    ) -> Iterator[str]:
        """Synchronous streaming — yields text chunks."""
        messages = self._to_messages(prompt)
        yield from self.provider.stream(messages, **self._merge_kwargs(**kwargs))

    def run(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        tools_only: bool = False,
        exec_code: bool = False,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Run a loop until the LLM produces a final text response.

        By default, runs in hybrid mode — the LLM has both tools and code
        execution available and decides which to use per step.

        orchestration — list of prompt fragment keys to stack on top of the
                        mode's base prompts. e.g. ["planning"] or
                        ["planning", "safety"]. Merged with
                        self.default_orchestration (default: ["repair"]).
                        Pass an empty list to disable defaults.
        tools_only — when True, only JSON tool calls are used (no code execution).
        exec_code — when True, only code execution is used (no tools).
        on_tool_call(ToolCall) — optional callback when LLM requests a tool
        on_tool_result(ToolResult) — optional callback after tool execution
        on_code_exec(str) — optional callback when code is about to run
        on_code_result(ExecResult) — optional callback after code execution

        Returns the final Response (the one with no tool calls / code blocks).
        """
        # Merge default + per-call orchestration, deduplicated, order preserved
        if orchestration is not None:
            combined = list(orchestration)
        else:
            combined = []
        for key in self.default_orchestration:
            if key not in combined:
                combined.append(key)
        orch = combined

        # Code-only mode
        if exec_code:
            return self._run_code_loop(
                prompt,
                orchestration=orch,
                on_code_exec=on_code_exec,
                on_code_result=on_code_result,
                **kwargs,
            )

        # Tool-only mode
        if tools_only:
            if not self.registry:
                return self.complete(prompt, **kwargs)
            return self._run_tool_loop(
                prompt,
                orchestration=orch,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                **kwargs,
            )

        # Default: hybrid mode (tools + code)
        if self.registry:
            return self._run_hybrid_loop(
                prompt,
                orchestration=orch,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_code_exec=on_code_exec,
                on_code_result=on_code_result,
                **kwargs,
            )

        # No tools registered — fall back to code-only
        return self._run_code_loop(
            prompt,
            orchestration=orch,
            on_code_exec=on_code_exec,
            on_code_result=on_code_result,
            **kwargs,
        )

    def _run_tool_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Run a tool-only loop (no code execution)."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)

        keys = list(orchestration or [])
        if keys:
            merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="tool")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=merged.get("tools"),
            )
            response = self.provider.complete(messages, **merged)
            hlog.log_llm_response(response)

            if not response.tool_calls:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            observations = []
            for tc in response.tool_calls:
                hlog.log_tool_call(tc)
                if on_tool_call:
                    on_tool_call(tc)

                result = self.registry.execute(tc)

                hlog.log_tool_result(result)
                if on_tool_result:
                    on_tool_result(result)

                messages.append(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                ))
                observations.append({
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                })

            traj.record_step(
                state=list(messages),
                action={
                    "type": "tool_calls",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                },
                observation={"results": observations},
                model=response.model,
                usage=response.usage,
            )

        traj.finish()
        return response

    def _run_code_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Execute a code-execution loop (LLMVM-style)."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)
        merged.pop("tools", None)

        keys = ["code_exec"] + list(orchestration or [])
        merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="code")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=None,
            )
            response = self.provider.complete(messages, **merged)
            hlog.log_llm_response(response)

            code_blocks = extract_code_blocks(response.content)

            if not code_blocks:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            messages.append(Message(role="assistant", content=response.content))

            exec_observations = []
            for code in code_blocks:
                if on_code_exec:
                    on_code_exec(code)

                exec_result = execute_code(code)

                if on_code_result:
                    on_code_result(exec_result)

                exec_observations.append({
                    "code": code,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "returncode": exec_result.returncode,
                })

            all_output = [obs["stdout"] or obs["stderr"] or "(no output)"
                          for obs in exec_observations]
            output_text = "\n---\n".join(all_output)
            messages.append(Message(
                role="user",
                content=f"Code execution output:\n{output_text}",
            ))

            traj.record_step(
                state=list(messages),
                action={"type": "code", "code_blocks": [o["code"] for o in exec_observations]},
                observation={"executions": exec_observations},
                model=response.model,
                usage=response.usage,
            )

        traj.finish()
        return response

    def _run_hybrid_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Run a hybrid loop where the LLM can use tools or code each step."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)

        keys = ["hybrid", "code_exec"] + list(orchestration or [])
        merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="hybrid")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=merged.get("tools"),
            )
            response = self.provider.complete(messages, **merged)
            hlog.log_llm_response(response)

            has_tool_calls = bool(response.tool_calls)
            code_blocks = extract_code_blocks(response.content)
            has_code = bool(code_blocks)

            # No actions — final text response
            if not has_tool_calls and not has_code:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            # Tool calls take priority (structured, from the API)
            if has_tool_calls:
                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                observations = []
                for tc in response.tool_calls:
                    hlog.log_tool_call(tc)
                    if on_tool_call:
                        on_tool_call(tc)

                    result = self.registry.execute(tc)

                    hlog.log_tool_result(result)
                    if on_tool_result:
                        on_tool_result(result)

                    messages.append(Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    ))
                    observations.append({
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    })

                traj.record_step(
                    state=list(messages),
                    action={
                        "type": "tool_calls",
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ],
                    },
                    observation={"results": observations},
                    model=response.model,
                    usage=response.usage,
                )

            # Code blocks (only if no tool calls this round)
            elif has_code:
                messages.append(Message(role="assistant", content=response.content))

                exec_observations = []
                for code in code_blocks:
                    if on_code_exec:
                        on_code_exec(code)

                    exec_result = execute_code(code)

                    if on_code_result:
                        on_code_result(exec_result)

                    exec_observations.append({
                        "code": code,
                        "stdout": exec_result.stdout,
                        "stderr": exec_result.stderr,
                        "returncode": exec_result.returncode,
                    })

                all_output = [obs["stdout"] or obs["stderr"] or "(no output)"
                              for obs in exec_observations]
                output_text = "\n---\n".join(all_output)
                messages.append(Message(
                    role="user",
                    content=f"Code execution output:\n{output_text}",
                ))

                traj.record_step(
                    state=list(messages),
                    action={"type": "code", "code_blocks": [o["code"] for o in exec_observations]},
                    observation={"executions": exec_observations},
                    model=response.model,
                    usage=response.usage,
                )

        traj.finish()
        return response

    # ── Async ───────────────────────────────────────────────────────

    async def acomplete(
        self,
        prompt: str | list[Message],
        **kwargs,
    ) -> Response:
        """Async one-shot completion."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(**kwargs)
        hlog.log_llm_request(
            messages,
            model=merged.get("model"),
            system=merged.get("system"),
            temperature=merged.get("temperature", 0.0),
            max_tokens=merged.get("max_tokens", 4096),
            tools=merged.get("tools"),
        )
        response = await self.provider.acomplete(messages, **merged)
        hlog.log_llm_response(response)
        return response

    async def astream(
        self,
        prompt: str | list[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async streaming — yields text chunks."""
        messages = self._to_messages(prompt)
        async for chunk in self.provider.astream(
            messages, **self._merge_kwargs(**kwargs)
        ):
            yield chunk

    async def arun(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        tools_only: bool = False,
        exec_code: bool = False,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Async version of run() — defaults to hybrid, with tool-only and code-only overrides."""
        if orchestration is not None:
            combined = list(orchestration)
        else:
            combined = []
        for key in self.default_orchestration:
            if key not in combined:
                combined.append(key)
        orch = combined

        if exec_code:
            return await self._arun_code_loop(
                prompt,
                orchestration=orch,
                on_code_exec=on_code_exec,
                on_code_result=on_code_result,
                **kwargs,
            )

        if tools_only:
            if not self.registry:
                return await self.acomplete(prompt, **kwargs)
            return await self._arun_tool_loop(
                prompt,
                orchestration=orch,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                **kwargs,
            )

        # Default: hybrid
        if self.registry:
            return await self._arun_hybrid_loop(
                prompt,
                orchestration=orch,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_code_exec=on_code_exec,
                on_code_result=on_code_result,
                **kwargs,
            )

        return await self._arun_code_loop(
            prompt,
            orchestration=orch,
            on_code_exec=on_code_exec,
            on_code_result=on_code_result,
            **kwargs,
        )

    async def _arun_tool_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Async version of _run_tool_loop."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)

        keys = list(orchestration or [])
        if keys:
            merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="tool")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=merged.get("tools"),
            )
            response = await self.provider.acomplete(messages, **merged)
            hlog.log_llm_response(response)

            if not response.tool_calls:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            observations = []
            for tc in response.tool_calls:
                hlog.log_tool_call(tc)
                if on_tool_call:
                    on_tool_call(tc)

                result = self.registry.execute(tc)

                hlog.log_tool_result(result)
                if on_tool_result:
                    on_tool_result(result)

                messages.append(Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                ))
                observations.append({
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                })

            traj.record_step(
                state=list(messages),
                action={
                    "type": "tool_calls",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                },
                observation={"results": observations},
                model=response.model,
                usage=response.usage,
            )

        traj.finish()
        return response

    async def _arun_code_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Async version of _run_code_loop."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)
        merged.pop("tools", None)

        keys = ["code_exec"] + list(orchestration or [])
        merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="code")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=None,
            )
            response = await self.provider.acomplete(messages, **merged)
            hlog.log_llm_response(response)

            code_blocks = extract_code_blocks(response.content)

            if not code_blocks:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            messages.append(Message(role="assistant", content=response.content))

            exec_observations = []
            for code in code_blocks:
                if on_code_exec:
                    on_code_exec(code)

                exec_result = execute_code(code)

                if on_code_result:
                    on_code_result(exec_result)

                exec_observations.append({
                    "code": code,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "returncode": exec_result.returncode,
                })

            all_output = [obs["stdout"] or obs["stderr"] or "(no output)"
                          for obs in exec_observations]
            output_text = "\n---\n".join(all_output)
            messages.append(Message(
                role="user",
                content=f"Code execution output:\n{output_text}",
            ))

            traj.record_step(
                state=list(messages),
                action={"type": "code", "code_blocks": [o["code"] for o in exec_observations]},
                observation={"executions": exec_observations},
                model=response.model,
                usage=response.usage,
            )

        traj.finish()
        return response

    async def _arun_hybrid_loop(
        self,
        prompt: str | list[Message],
        *,
        orchestration: list[str] | None = None,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        on_code_exec: callable | None = None,
        on_code_result: callable | None = None,
        **kwargs,
    ) -> Response:
        """Async version of _run_hybrid_loop."""
        messages = self._to_messages(prompt)
        merged = self._merge_kwargs(orchestration=orchestration, **kwargs)

        keys = ["hybrid", "code_exec"] + list(orchestration or [])
        merged["system"] = self._build_system_prompt(keys, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else messages[-1].content
        traj = Trajectory(prompt_text, mode="hybrid")

        for round_num in range(MAX_TOOL_ROUNDS):
            hlog.log_llm_request(
                messages,
                model=merged.get("model"),
                system=merged.get("system"),
                temperature=merged.get("temperature", 0.0),
                max_tokens=merged.get("max_tokens", 4096),
                tools=merged.get("tools"),
            )
            response = await self.provider.acomplete(messages, **merged)
            hlog.log_llm_response(response)

            has_tool_calls = bool(response.tool_calls)
            code_blocks = extract_code_blocks(response.content)
            has_code = bool(code_blocks)

            if not has_tool_calls and not has_code:
                traj.record_step(
                    state=list(messages),
                    action={"type": "text", "content": response.content},
                    model=response.model,
                    usage=response.usage,
                )
                traj.finish()
                return response

            if has_tool_calls:
                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                observations = []
                for tc in response.tool_calls:
                    hlog.log_tool_call(tc)
                    if on_tool_call:
                        on_tool_call(tc)

                    result = self.registry.execute(tc)

                    hlog.log_tool_result(result)
                    if on_tool_result:
                        on_tool_result(result)

                    messages.append(Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    ))
                    observations.append({
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    })

                traj.record_step(
                    state=list(messages),
                    action={
                        "type": "tool_calls",
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ],
                    },
                    observation={"results": observations},
                    model=response.model,
                    usage=response.usage,
                )

            elif has_code:
                messages.append(Message(role="assistant", content=response.content))

                exec_observations = []
                for code in code_blocks:
                    if on_code_exec:
                        on_code_exec(code)

                    exec_result = execute_code(code)

                    if on_code_result:
                        on_code_result(exec_result)

                    exec_observations.append({
                        "code": code,
                        "stdout": exec_result.stdout,
                        "stderr": exec_result.stderr,
                        "returncode": exec_result.returncode,
                    })

                all_output = [obs["stdout"] or obs["stderr"] or "(no output)"
                              for obs in exec_observations]
                output_text = "\n---\n".join(all_output)
                messages.append(Message(
                    role="user",
                    content=f"Code execution output:\n{output_text}",
                ))

                traj.record_step(
                    state=list(messages),
                    action={"type": "code", "code_blocks": [o["code"] for o in exec_observations]},
                    observation={"executions": exec_observations},
                    model=response.model,
                    usage=response.usage,
                )

        traj.finish()
        return response
