"""Evaluation framework for comparing execution configurations.

Runs the same prompt through multiple configurations (mode + orchestration
combos), captures trajectory data, and grades the results.

    runner = EvalRunner(harness, configs=[
        EvalConfig("hybrid"),
        EvalConfig("hybrid", orchestration=["planning"]),
        EvalConfig("tool"),
        EvalConfig("code"),
    ])
    results = runner.run_suite(cases)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .core import Harness
from .types import Message, Response, ToolCall, ToolResult
from .executor import ExecResult


class Mode(str, Enum):
    TOOL = "tool"
    CODE = "code"
    HYBRID = "hybrid"


class Grade(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True)
class EvalConfig:
    """A configuration to evaluate — mode + optional orchestration.

        EvalConfig("hybrid")
        EvalConfig("hybrid", orchestration=["planning"])
        EvalConfig("tool")
        EvalConfig("code", orchestration=["concise"])
    """

    mode: str = "hybrid"
    orchestration: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short label for display: 'hybrid', 'tool+planning', etc."""
        parts = [self.mode]
        parts.extend(self.orchestration)
        return "+".join(parts)


# Default configs: the three base modes
DEFAULT_CONFIGS = [
    EvalConfig("tool"),
    EvalConfig("code"),
    EvalConfig("hybrid"),
]


@dataclass
class EvalCase:
    """A single evaluation case.

    name — short identifier
    prompt — the task to run
    preferred_mode — which mode a human would pick (for scoring)
    check — optional callable(RunResult) -> bool for task-specific validation
    setup — optional callable() to run before the eval
    teardown — optional callable() to run after the eval
    tags — for filtering (e.g. "file_io", "computation")
    """

    name: str
    prompt: str
    preferred_mode: Mode = Mode.HYBRID
    check: Callable[[RunResult], bool] | None = None
    setup: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of running one eval case in one configuration."""

    config: EvalConfig
    response: Response
    rounds: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class EvalResult:
    """Result of running one eval case through all configs."""

    case: EvalCase
    runs: dict[str, RunResult] = field(default_factory=dict)
    grades: dict[str, Grade] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        """Flat dict for tabular/JSON output."""
        d: dict[str, Any] = {
            "name": self.case.name,
            "preferred": self.case.preferred_mode.value,
            "tags": ",".join(self.case.tags),
        }
        for label, run in self.runs.items():
            grade = self.grades.get(label, Grade.ERROR)
            d[f"{label}_grade"] = grade.value if isinstance(grade, Grade) else str(grade)
            d[f"{label}_rounds"] = run.rounds
            d[f"{label}_tokens"] = run.total_input_tokens + run.total_output_tokens
            d[f"{label}_errors"] = len(run.errors)
            d[f"{label}_duration"] = round(run.duration_s, 2)
            if run.tool_calls:
                d[f"{label}_tool_calls"] = len(run.tool_calls)
            if run.code_blocks:
                d[f"{label}_code_blocks"] = len(run.code_blocks)
        d["notes"] = "; ".join(self.notes)
        return d


class EvalRunner:
    """Runs eval cases through a harness across multiple configurations."""

    def __init__(
        self,
        harness: Harness,
        *,
        configs: list[EvalConfig] | None = None,
        verbose: bool = False,
    ):
        self.harness = harness
        self.configs = configs or DEFAULT_CONFIGS
        self.verbose = verbose

    def run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case through all configured modes."""
        result = EvalResult(case=case)

        if case.setup:
            case.setup()

        try:
            for config in self.configs:
                if self.verbose:
                    print(f"  [{case.name}] running {config.label}...")
                run = self._run_config(case, config)
                grade = self._grade(case, run)
                result.runs[config.label] = run
                result.grades[config.label] = grade

            result.notes = self._compare(result)
        finally:
            if case.teardown:
                case.teardown()

        return result

    def run_suite(self, cases: list[EvalCase], *, tags: list[str] | None = None) -> list[EvalResult]:
        """Run all cases (optionally filtered by tags)."""
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]

        results = []
        for i, case in enumerate(cases, 1):
            if self.verbose:
                print(f"[{i}/{len(cases)}] {case.name}")
            results.append(self.run_case(case))

        return results

    def _run_config(self, case: EvalCase, config: EvalConfig) -> RunResult:
        """Run a case with a specific configuration."""
        tool_calls_log: list[dict[str, Any]] = []
        code_blocks_log: list[str] = []
        errors: list[str] = []
        rounds = 0

        def on_tool_call(tc: ToolCall):
            tool_calls_log.append({"name": tc.name, "arguments": tc.arguments})

        def on_tool_result(tr: ToolResult):
            nonlocal rounds
            rounds += 1
            if tr.is_error:
                errors.append(f"tool error ({tr.tool_call_id}): {tr.content[:200]}")

        def on_code_exec(code: str):
            code_blocks_log.append(code)

        def on_code_result(er: ExecResult):
            nonlocal rounds
            rounds += 1
            if not er.ok:
                errors.append(f"code error (rc={er.returncode}): {er.stderr[:200]}")

        orch = config.orchestration or None

        start = time.time()
        try:
            if config.mode == "code":
                response = self.harness.run(
                    case.prompt,
                    exec_code=True,
                    orchestration=orch,
                    on_code_exec=on_code_exec,
                    on_code_result=on_code_result,
                )
            elif config.mode == "tool":
                response = self.harness.run(
                    case.prompt,
                    tools_only=True,
                    orchestration=orch,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                )
            else:
                response = self.harness.run(
                    case.prompt,
                    orchestration=orch,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_code_exec=on_code_exec,
                    on_code_result=on_code_result,
                )
        except Exception as e:
            errors.append(f"exception: {type(e).__name__}: {e}")
            if self.verbose:
                print(f"    ERROR: {type(e).__name__}: {e}")
            response = Response(content="", model="error", stop_reason="error")

        duration = time.time() - start

        return RunResult(
            config=config,
            response=response,
            rounds=rounds,
            tool_calls=tool_calls_log,
            code_blocks=code_blocks_log,
            errors=errors,
            duration_s=duration,
            total_input_tokens=response.usage.get("input", 0),
            total_output_tokens=response.usage.get("output", 0),
        )

    def _grade(self, case: EvalCase, run: RunResult) -> Grade:
        """Grade a single run result."""
        if any("exception:" in e for e in run.errors):
            return Grade.ERROR

        if case.check:
            try:
                return Grade.PASS if case.check(run) else Grade.FAIL
            except Exception as e:
                if self.verbose:
                    print(f"    Check raised: {type(e).__name__}: {e}")
                return Grade.ERROR

        if not run.response.content:
            return Grade.FAIL
        if run.has_errors:
            return Grade.PARTIAL
        return Grade.PASS

    def _compare(self, result: EvalResult) -> list[str]:
        """Generate comparison notes across all configs."""
        notes = []
        runs = result.runs
        grades = result.grades

        if len(runs) < 2:
            return notes

        # Find the best config by grade, then rounds, then duration
        passing = {k: v for k, v in runs.items()
                   if grades.get(k) in (Grade.PASS, Grade.PARTIAL)}

        if not passing:
            notes.append("all configs failed")
            return notes

        best = min(passing.keys(), key=lambda k: (
            0 if grades[k] == Grade.PASS else 1,
            len(passing[k].errors),
            passing[k].rounds,
            passing[k].duration_s,
        ))
        notes.append(f"best: {best}")

        # Per-config summaries
        for label, run in runs.items():
            grade = grades.get(label, Grade.ERROR)
            parts = [f"{grade.value}, {run.rounds}r, {run.duration_s:.1f}s"]
            if run.tool_calls:
                parts.append(f"{len(run.tool_calls)} tools")
            if run.code_blocks:
                parts.append(f"{len(run.code_blocks)} code")
            notes.append(f"  {label}: {', '.join(parts)}")

        return notes


def print_report(results: list[EvalResult]) -> None:
    """Print a human-readable comparison report."""
    if not results:
        print("No results.")
        return

    # Collect all config labels from the results
    all_labels = []
    for r in results:
        for label in r.runs:
            if label not in all_labels:
                all_labels.append(label)

    print()
    print("=" * 80)
    print("EVAL REPORT")
    print("=" * 80)

    # Summary counts per config
    total = len(results)
    print(f"\nTotal cases: {total}")
    for label in all_labels:
        p = sum(1 for r in results if r.grades.get(label) == Grade.PASS)
        f = sum(1 for r in results if r.grades.get(label) == Grade.FAIL)
        pa = sum(1 for r in results if r.grades.get(label) == Grade.PARTIAL)
        e = sum(1 for r in results if r.grades.get(label) == Grade.ERROR)
        print(f"  {label:<25} {p} pass, {pa} partial, {f} fail, {e} error")

    # Table
    col_w = max(10, max(len(l) for l in all_labels) + 2)
    header = f"{'Case':<30}"
    for label in all_labels:
        header += f" {label:>{col_w}}"
    header += f" {'Best':>{col_w}}"

    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for r in results:
        row = f"{r.case.name:<30}"
        for label in all_labels:
            grade = r.grades.get(label, Grade.ERROR).value
            row += f" {grade:>{col_w}}"
        # Best from notes
        best = "—"
        for n in r.notes:
            if n.startswith("best: "):
                best = n[6:]
                break
        row += f" {best:>{col_w}}"
        print(row)

    # Details
    print(f"\n{'=' * 80}")
    print("DETAILS")
    print(f"{'=' * 80}")

    for r in results:
        print(f"\n── {r.case.name} ──")
        print(f"  Prompt: {r.case.prompt[:100]}{'...' if len(r.case.prompt) > 100 else ''}")
        print(f"  Tags: {', '.join(r.case.tags)}")

        for label, run in r.runs.items():
            grade = r.grades.get(label, Grade.ERROR)
            tokens = run.total_input_tokens + run.total_output_tokens
            print(f"  {label}: {grade.value} | {run.rounds} rounds | "
                  f"{tokens} tokens | {run.duration_s:.1f}s")
            if run.tool_calls:
                tools_used = [tc["name"] for tc in run.tool_calls]
                print(f"    Tools: {', '.join(tools_used)}")
            if run.code_blocks:
                print(f"    Code blocks: {len(run.code_blocks)}")
            if run.errors:
                for e in run.errors[:3]:
                    print(f"    Error: {e}")

        for note in r.notes:
            print(f"  * {note}")

    print()


def load_cases_from_json(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file.

    JSON format:
        [
            {"name": "...", "prompt": "...", "preferred_mode": "tool", "tags": ["file_io"]},
            ...
        ]

    Note: check, setup, and teardown cannot be specified in JSON.
    """
    data = json.loads(Path(path).read_text())
    cases = []
    for item in data:
        cases.append(EvalCase(
            name=item["name"],
            prompt=item["prompt"],
            preferred_mode=Mode(item.get("preferred_mode", "hybrid")),
            tags=item.get("tags", []),
        ))
    return cases


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """Save eval results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_cases": len(results),
        "results": [r.summary_dict() for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")
