"""Eval suite for comparing execution configurations.

15 built-in cases across three categories. Supports custom configs
to compare any combination of mode + orchestration.

Run:
    python eval_suite.py                                    # default: tool vs code vs hybrid
    python eval_suite.py --configs hybrid hybrid+planning   # compare hybrid with and without planning
    python eval_suite.py --tags ambiguous --verbose          # just ambiguous cases
    python eval_suite.py --json evals/my_cases.json          # load cases from JSON
    python eval_suite.py -m llama3.2                         # different model
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from byoh import Harness
from byoh.eval import (
    EvalCase, EvalConfig, EvalRunner, Mode, RunResult,
    load_cases_from_json, print_report, save_results,
)
from byoh.tools import FILE_TOOLS


# ── Temp file helpers for eval setup/teardown ────────────────────────

_temp_dir = Path(tempfile.mkdtemp(prefix="byoh_eval_"))


def _create_sample_file():
    """Create a sample file for read-based evals."""
    p = _temp_dir / "sample.txt"
    p.write_text("line 1: hello world\nline 2: foo bar\nline 3: baz qux\n")


def _create_csv_file():
    """Create a CSV file for data processing evals."""
    p = _temp_dir / "data.csv"
    p.write_text(
        "name,age,score\n"
        "alice,30,85\n"
        "bob,25,92\n"
        "carol,35,78\n"
        "dave,28,95\n"
        "eve,32,88\n"
    )


def _create_json_file():
    """Create a JSON file for parsing evals."""
    import json
    p = _temp_dir / "config.json"
    p.write_text(json.dumps({
        "database": {"host": "localhost", "port": 5432, "name": "mydb"},
        "cache": {"ttl": 300, "max_size": 1000},
        "features": ["auth", "logging", "metrics"],
    }, indent=2))


def _create_python_file():
    """Create a Python file for code analysis evals."""
    p = _temp_dir / "example.py"
    p.write_text(
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        "\n"
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    )


def _cleanup():
    """Remove temp files."""
    import shutil
    shutil.rmtree(_temp_dir, ignore_errors=True)


# ── Custom check functions ───────────────────────────────────────────

def _check_has_content(run: RunResult) -> bool:
    """Pass if the response has non-empty content."""
    return bool(run.response.content.strip())


def _check_mentions_lines(run: RunResult) -> bool:
    """Pass if the response mentions the file content."""
    text = run.response.content.lower()
    return "hello world" in text or "foo bar" in text


def _check_has_number(run: RunResult) -> bool:
    """Pass if the response contains a numeric result."""
    import re
    return bool(re.search(r'\d+', run.response.content))


def _check_mentions_fibonacci(run: RunResult) -> bool:
    """Pass if the response correctly identifies the fibonacci function."""
    text = run.response.content.lower()
    return "fibonacci" in text


def _check_file_written(run: RunResult) -> bool:
    """Pass if the output file was created."""
    return (_temp_dir / "output.txt").exists()


def _check_csv_stats(run: RunResult) -> bool:
    """Pass if the response mentions reasonable statistics from the CSV."""
    text = run.response.content.lower()
    # Average age is 30, average score is 87.6
    return any(kw in text for kw in ["30", "87", "average", "mean"])


# ── Eval cases ───────────────────────────────────────────────────────

CASES = [
    # ── Tool-natural: tasks where structured tool calls are the right fit ──

    EvalCase(
        name="read_simple_file",
        prompt=f"Read the file at {_temp_dir}/sample.txt and tell me what's in it.",
        preferred_mode=Mode.TOOL,
        check=_check_mentions_lines,
        setup=_create_sample_file,
        tags=["tool_natural", "file_io"],
    ),
    EvalCase(
        name="write_simple_file",
        prompt=f"Write the text 'hello from eval' to {_temp_dir}/output.txt",
        preferred_mode=Mode.TOOL,
        check=_check_file_written,
        tags=["tool_natural", "file_io"],
    ),
    EvalCase(
        name="read_and_summarize",
        prompt=f"Read {_temp_dir}/config.json and list what features are configured.",
        preferred_mode=Mode.TOOL,
        setup=_create_json_file,
        check=lambda r: "auth" in r.response.content.lower(),
        tags=["tool_natural", "file_io"],
    ),
    EvalCase(
        name="copy_file_content",
        prompt=f"Read {_temp_dir}/sample.txt and write its content to {_temp_dir}/copy.txt",
        preferred_mode=Mode.TOOL,
        setup=_create_sample_file,
        check=lambda r: (_temp_dir / "copy.txt").exists(),
        tags=["tool_natural", "file_io"],
    ),
    EvalCase(
        name="read_python_describe",
        prompt=f"Read {_temp_dir}/example.py and list the function names defined in it.",
        preferred_mode=Mode.TOOL,
        setup=_create_python_file,
        check=_check_mentions_fibonacci,
        tags=["tool_natural", "file_io", "code_understanding"],
    ),

    # ── Code-natural: tasks where Python execution is the right fit ──

    EvalCase(
        name="compute_fibonacci",
        prompt="Calculate the 20th Fibonacci number and tell me the result.",
        preferred_mode=Mode.CODE,
        check=lambda r: "6765" in r.response.content,
        tags=["code_natural", "computation"],
    ),
    EvalCase(
        name="generate_primes",
        prompt="Generate all prime numbers below 50 and list them.",
        preferred_mode=Mode.CODE,
        check=lambda r: "47" in r.response.content and "2" in r.response.content,
        tags=["code_natural", "computation"],
    ),
    EvalCase(
        name="string_manipulation",
        prompt="Take the string 'hello world' and output it reversed, then in uppercase, then count the vowels.",
        preferred_mode=Mode.CODE,
        check=lambda r: "dlrow" in r.response.content.lower() or "HELLO" in r.response.content,
        tags=["code_natural", "computation"],
    ),
    EvalCase(
        name="sort_numbers",
        prompt="Sort these numbers in descending order: 42, 17, 93, 8, 55, 31, 76. Output just the sorted list.",
        preferred_mode=Mode.CODE,
        check=lambda r: "93" in r.response.content,
        tags=["code_natural", "computation"],
    ),
    EvalCase(
        name="math_expression",
        prompt="What is (17 * 23) + (45 / 9) - (12 ** 2)? Show the exact result.",
        preferred_mode=Mode.CODE,
        check=lambda r: "252" in r.response.content or "252.0" in r.response.content,
        tags=["code_natural", "computation"],
    ),

    # ── Ambiguous: tasks where either mode could work ──

    EvalCase(
        name="csv_statistics",
        prompt=f"Read the CSV file at {_temp_dir}/data.csv and compute the average age and average score.",
        preferred_mode=Mode.CODE,
        setup=_create_csv_file,
        check=_check_csv_stats,
        tags=["ambiguous", "file_io", "computation"],
    ),
    EvalCase(
        name="file_line_count",
        prompt=f"Count the number of lines in {_temp_dir}/sample.txt",
        preferred_mode=Mode.TOOL,
        setup=_create_sample_file,
        check=lambda r: "3" in r.response.content,
        tags=["ambiguous", "file_io"],
    ),
    EvalCase(
        name="json_transform",
        prompt=(
            f"Read {_temp_dir}/config.json, add a new feature called 'notifications' "
            f"to the features list, and write the updated JSON back to {_temp_dir}/config_updated.json"
        ),
        preferred_mode=Mode.CODE,
        setup=_create_json_file,
        check=lambda r: (_temp_dir / "config_updated.json").exists(),
        tags=["ambiguous", "file_io", "computation"],
    ),
    EvalCase(
        name="python_refactor_suggestion",
        prompt=(
            f"Read {_temp_dir}/example.py. The fibonacci function uses naive recursion. "
            f"Suggest an improved version and write it to {_temp_dir}/example_improved.py"
        ),
        preferred_mode=Mode.CODE,
        setup=_create_python_file,
        check=lambda r: (_temp_dir / "example_improved.py").exists(),
        tags=["ambiguous", "file_io", "code_understanding"],
    ),
    EvalCase(
        name="generate_and_save",
        prompt=f"Generate a multiplication table (1-5) and save it to {_temp_dir}/table.txt",
        preferred_mode=Mode.CODE,
        check=lambda r: (_temp_dir / "table.txt").exists(),
        tags=["ambiguous", "computation", "file_io"],
    ),
]


# ── Config parsing ───────────────────────────────────────────────────

def _parse_config(spec: str) -> EvalConfig:
    """Parse a config spec like 'hybrid', 'hybrid+planning', 'tool+safety+concise'."""
    parts = spec.split("+")
    mode = parts[0]
    orchestration = parts[1:] if len(parts) > 1 else []
    return EvalConfig(mode=mode, orchestration=orchestration)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run byoh eval suite — compare execution configurations"
    )
    parser.add_argument("-m", "--model", default=None, help="model override")
    parser.add_argument("--configs", nargs="*", default=None, metavar="SPEC",
                        help="configs to compare (e.g. hybrid hybrid+planning tool code). "
                             "Format: mode or mode+orch1+orch2. Default: tool, code, hybrid")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="filter by tags (e.g. tool_natural code_natural ambiguous)")
    parser.add_argument("--cases", nargs="*", default=None, help="run specific cases by name")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="load additional cases from a JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress")
    parser.add_argument("-o", "--output", default=None, help="save results JSON to this path")
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    args = parser.parse_args()

    # Load cases
    cases = list(CASES)
    if args.json:
        cases.extend(load_cases_from_json(args.json))

    if args.list:
        print(f"{'Name':<30} {'Preferred':>8} {'Tags'}")
        print("─" * 70)
        for c in cases:
            print(f"{c.name:<30} {c.preferred_mode.value:>8} {', '.join(c.tags)}")
        return

    # Parse configs
    configs = None
    if args.configs:
        configs = [_parse_config(spec) for spec in args.configs]

    harness = Harness(model=args.model, tools=FILE_TOOLS)

    # Filter cases by name
    if args.cases:
        cases = [c for c in cases if c.name in args.cases]
        if not cases:
            sys.exit(f"No matching cases found. Use --list to see available cases.")

    runner = EvalRunner(harness, configs=configs, verbose=args.verbose)

    config_labels = [c.label for c in runner.configs]
    print(f"Running {len(cases)} eval cases × {len(runner.configs)} configs: {', '.join(config_labels)}")
    print(f"Temp dir: {_temp_dir}")
    print()

    try:
        results = runner.run_suite(cases, tags=args.tags)
        print_report(results)

        if args.output:
            save_results(results, args.output)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
