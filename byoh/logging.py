"""Structured logging for byoh operations.

Two log streams:

1. byoh.log — human-readable, truncated, for tailing in real time.
   One JSON line per event. Configurable via BYOH_LOG env var.

2. trajectories/ — full RL-ready trajectory files, one per run().
   Each file is a JSON array of steps with untruncated data, run_id,
   step index, full message state, actions, and observations.
   Configurable via BYOH_TRAJECTORY_DIR env var.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .types import Message, Response, Tool, ToolCall, ToolResult


# Where logs go — override with env vars
DEFAULT_LOG_DIR = Path.home() / ".byoh"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "byoh.log"
DEFAULT_TRAJECTORY_DIR = DEFAULT_LOG_DIR / "trajectories"

# Truncate long strings in human-readable logs
MAX_CONTENT_LEN = 500


def _truncate(s: str, max_len: int = MAX_CONTENT_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... ({len(s)} chars total)"


def _message_to_dict(m: Message, truncate: bool = True) -> dict:
    content = _truncate(m.content) if truncate else m.content
    d: dict[str, Any] = {"role": m.role, "content": content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


# ── Human-readable log (truncated, for tailing) ───────────────────

def setup_logger() -> logging.Logger:
    """Create and configure the harness file logger."""
    log_path = Path(os.environ.get("BYOH_LOG", DEFAULT_LOG_FILE))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("byoh")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger


_logger = setup_logger()


def _emit(event: str, **data: Any) -> None:
    """Write a single JSON log line to harness.log."""
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **data,
    }
    _logger.info(json.dumps(entry, default=str))


def log_llm_request(
    messages: list[Message],
    *,
    model: str | None,
    system: str | None,
    temperature: float,
    max_tokens: int,
    tools: list[Tool] | None,
) -> None:
    _emit(
        "llm_request",
        model=model,
        system=_truncate(system) if system else None,
        temperature=temperature,
        max_tokens=max_tokens,
        message_count=len(messages),
        messages=[_message_to_dict(m) for m in messages[-5:]],
        tools=[t.name for t in tools] if tools else None,
    )


def log_llm_response(response: Response) -> None:
    _emit(
        "llm_response",
        model=response.model,
        content=_truncate(response.content),
        stop_reason=response.stop_reason,
        usage=response.usage,
        tool_calls=[
            {"name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ] if response.tool_calls else None,
    )


def log_tool_call(tc: ToolCall) -> None:
    _emit(
        "tool_call",
        tool_call_id=tc.id,
        name=tc.name,
        arguments=tc.arguments,
    )


def log_tool_result(result: ToolResult) -> None:
    _emit(
        "tool_result",
        tool_call_id=result.tool_call_id,
        content=_truncate(result.content),
        is_error=result.is_error,
    )


# ── RL Trajectory logging (untruncated, full state) ───────────────

class Trajectory:
    """Records a full RL-ready trajectory for one run() invocation.

    Each step captures:
      - step index
      - state: full message history at decision time (untruncated)
      - action: what the LLM chose (tool calls, code blocks, or final text)
      - observation: results fed back (tool results, code output)
      - metadata: model, usage, timing

    The trajectory is written to a JSON file when finish() is called.
    """

    def __init__(self, prompt: str, *, mode: str = "tool"):
        self.run_id = uuid.uuid4().hex[:12]
        self.mode = mode  # "tool" | "code"
        self.prompt = prompt
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.time()
        self._step_counter = 0
        self.reward: float | None = None

    def record_step(
        self,
        *,
        state: list[Message],
        action: dict[str, Any],
        observation: dict[str, Any] | None = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Record one step of the trajectory."""
        self.steps.append({
            "step": self._step_counter,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": [_message_to_dict(m, truncate=False) for m in state],
            "action": action,
            "observation": observation,
            "model": model,
            "usage": usage,
        })
        self._step_counter += 1

    def set_reward(self, reward: float) -> None:
        """Set the reward signal for this trajectory."""
        self.reward = reward

    def finish(self) -> Path:
        """Write the trajectory to disk and return the file path."""
        traj_dir = Path(os.environ.get(
            "BYOH_TRAJECTORY_DIR", DEFAULT_TRAJECTORY_DIR
        ))
        traj_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = traj_dir / f"{ts}_{self.run_id}.json"

        data = {
            "run_id": self.run_id,
            "mode": self.mode,
            "prompt": self.prompt,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration_s": round(time.time() - self.start_time, 2),
            "num_steps": len(self.steps),
            "reward": self.reward,
            "steps": self.steps,
        }

        path.write_text(json.dumps(data, indent=2, default=str))
        _emit("trajectory_saved", run_id=self.run_id, path=str(path),
              num_steps=len(self.steps), reward=self.reward)

        return path
