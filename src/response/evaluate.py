"""Sandboxed execution of stdin/stdout code against a record's test cases."""

import platform
import resource
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# the eval venv keeps a controlled, minimal set of libraries importable by
# generated code, kept separate from this project's own dependencies
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL_PYTHON = _REPO_ROOT / ".venv-eval" / "bin" / "python"

_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MEMORY_MB = 512


def _set_memory_limit(
    memory_mb: int,
) -> None:
    """
    Set the virtual memory limit for the current process (child only, via preexec_fn).

    Skipped on macOS - RLIMIT_AS is defined but enforcing it crashes python subprocesses.
    """
    if platform.system() == "Darwin":
        return
    try:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        # platform does not support RLIMIT_AS or the limit is invalid
        pass


def resolve_eval_python(
    python_executable: str | Path | None = None,
) -> Path:
    """
    Resolve the python interpreter to run generated code with.

    Defaults to the dedicated evaluation venv at .venv-eval, which holds a
    controlled set of libraries separate from this project's own dependencies.

    Returns the resolved interpreter path.
    """
    # resolve to absolute - execute_code runs the subprocess in a different cwd
    resolved = Path(python_executable).resolve() if python_executable else _EVAL_PYTHON
    if not resolved.exists():
        raise FileNotFoundError(
            f"evaluation python interpreter not found at {resolved}. "
            "Run `bash setup/setup_eval_venv.sh <requirements-version>` to create it.",
        )

    return resolved


def execute_code(
    code: str,
    python_executable: str | Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
) -> tuple[bool, str]:
    """
    Run code in a sandboxed subprocess with time and memory limits.

    Returns a (success, captured stdout or an error message) tuple.
    """
    python_executable = resolve_eval_python(python_executable=python_executable)

    try:
        # run in a temp dir so any files created by the code don't pollute the repo
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [str(python_executable), "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=lambda: _set_memory_limit(memory_mb),
                cwd=tmp_dir,
            )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"

    # success: return captured stdout
    if result.returncode == 0:
        return True, result.stdout

    # killed by a signal (negative return code on unix)
    if result.returncode < 0:
        sig_num = -result.returncode
        try:
            sig_name = signal.Signals(sig_num).name
        except (ValueError, AttributeError):
            sig_name = f"signal {sig_num}"
        return False, f"killed by {sig_name}"

    # normal failure - use stderr if available, otherwise stdout
    error = result.stderr.strip() or result.stdout.strip()
    return False, error or f"exit code {result.returncode}"


def _build_stdin_test_code(
    code: str,
    inputs: list[str],
    expected_outputs: list[str],
) -> str:
    """
    Build executable code that runs a candidate solution against stdin/stdout test cases.

    The candidate code is embedded as a string and exec'd once per test case with
    stdin/stdout mocked, since it's expected to read from stdin and print to stdout.

    Returns the complete python code to run via execute_code.
    """
    # escape the code for embedding in a triple-quoted string
    escaped_code = code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    test_cases = [
        f"({repr(inp)}, {repr(exp.strip() if isinstance(exp, str) else exp)})"
        for inp, exp in zip(inputs, expected_outputs)
    ]

    return f'''
import sys
from io import StringIO

_SOLUTION_CODE = """{escaped_code}"""


def _run_solution(stdin_input: str) -> str:
    """Run the solution code with mocked stdin and return its captured stdout."""
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = StringIO(stdin_input)
    captured = StringIO()
    sys.stdout = captured

    try:
        exec(_SOLUTION_CODE, {{"__name__": "__main__"}})
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout

    return captured.getvalue().strip()


_test_cases = [{", ".join(test_cases)}]
for _idx, (_input, _expected) in enumerate(_test_cases):
    _output = _run_solution(_input)
    assert _output == _expected, f"test {{_idx + 1}}: expected {{_expected!r}}, got {{_output!r}}"
'''


def evaluate_record(
    code: str,
    record: dict[str, Any],
    python_executable: str | Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """
    Evaluate a candidate code fragment against a stdin/stdout record's test cases.

    The record must have "inputs" and "expected_outputs" lists of equal length, as
    produced by data/download_skywork.py and data/download_eval.py.

    Returns a (passed, captured stdout or an error message) tuple.
    """
    test_code = _build_stdin_test_code(
        code=code,
        inputs=record["inputs"],
        expected_outputs=record["expected_outputs"],
    )
    return execute_code(
        code=test_code,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
