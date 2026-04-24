"""Code execution engine for LLMVM-style operation.

Instead of JSON tool calls, the LLM emits Python code in fenced blocks.
The harness extracts the code, runs it in a subprocess, and feeds
stdout/stderr back as the next message.

Uses subprocess for isolation — the code runs in a separate process
with a timeout, not exec() in the harness process.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import logging as hlog


# Match ```python ... ``` or ``` ... ``` code blocks
CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)

# Max time a code block can run before being killed
EXEC_TIMEOUT = 30


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of executing a code block."""

    code: str
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Combined output for feeding back to the LLM."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)


def extract_code_blocks(content: str) -> list[str]:
    """Extract Python code from fenced code blocks in LLM output."""
    return CODE_BLOCK_RE.findall(content)


def execute_code(code: str, *, timeout: int = EXEC_TIMEOUT, cwd: str | None = None) -> ExecResult:
    """Run a Python code string in a subprocess.

    The code is written to a temp file and executed with the same
    Python interpreter running the harness.
    """

    hlog._emit("code_exec", code=hlog._truncate(code, 1000))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        exec_result = ExecResult(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        exec_result = ExecResult(
            code=code,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            returncode=-1,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    hlog._emit(
        "code_result",
        returncode=exec_result.returncode,
        stdout=hlog._truncate(exec_result.stdout),
        stderr=hlog._truncate(exec_result.stderr),
    )

    return exec_result


# System prompt that tells the LLM to emit code instead of tool calls
CODE_EXEC_PROMPT = """You can execute Python code to accomplish tasks. Write code in fenced code blocks like:

```python
# your code here
print("result")
```

Rules:
- Use print() to output results — stdout is captured and fed back to you.
- You can read/write files, run shell commands via subprocess, install packages, etc.
- Each code block runs in a fresh process in the current working directory.
- If your code fails, you'll see the stderr and can fix it.
- When the task is done, respond with plain text (no code block) to finish.
- Be concise — write the minimum code needed."""
