"""Run candidate stdin/stdout programs, and manage the venv the harness installs into.

Two layers. `run_program` / `evaluate_record` execute a stdin/stdout program in a sandboxed
subprocess (time + memory limited) - used both to mint oracle outputs at build time and to score
answers. `Environment` wraps a cached virtual environment the harness installs a response's
libraries into (arbitrary, dynamic) before running it, so each answer is evaluated with exactly
the third-party packages it reached for.
"""

import platform
import subprocess
import sys
import tempfile
import venv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple


try:  # resource limits are posix-only
    import resource
except ImportError:  # pragma: no cover - windows
    resource = None  # type: ignore[assignment]


# generous enough for heavy scientific libraries (sklearn/scipy/statsmodels import slowly), so a
# genuinely dual-solvable task is not dropped just for import latency under a tight cap
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MEMORY_MB = 1024

# the interpreter used for build-time oracle runs (the project venv, which already has the
# reference solutions' libraries); the harness overrides this with its own venv's python
BUILD_PYTHON = sys.executable


def _set_memory_limit(
    memory_mb: int,
) -> None:
    """Cap the child process's virtual memory (posix only; skipped on macOS, where it crashes)."""
    if resource is None or platform.system() == "Darwin":
        return
    try:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass


def execute(
    code: str,
    python_executable: str | Path = BUILD_PYTHON,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
) -> tuple[bool, str]:
    """Run python source in a sandboxed subprocess with time and memory limits.

    Returns a (success, captured stdout or an error message) tuple.
    """
    try:
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

    if result.returncode == 0:
        return True, result.stdout
    error = result.stderr.strip() or result.stdout.strip()
    return False, error or f"exit code {result.returncode}"


def _capture_harness(
    program: str,
    stdin: str,
) -> str:
    """Wrap a stdin/stdout program to run once with mocked stdin and re-emit its stdout."""
    return (
        "import sys\n"
        "from io import StringIO\n"
        f"_SRC = {program!r}\n"
        f"_STDIN = {stdin!r}\n"
        "_old_in, _old_out = sys.stdin, sys.stdout\n"
        "sys.stdin = StringIO(_STDIN)\n"
        "_cap = StringIO()\n"
        "sys.stdout = _cap\n"
        "try:\n"
        '    exec(compile(_SRC, "<program>", "exec"), {"__name__": "__main__"})\n'
        "finally:\n"
        "    sys.stdin, sys.stdout = _old_in, _old_out\n"
        "_old_out.write(_cap.getvalue())\n"
    )


def _test_harness(
    program: str,
    inputs: list[str],
    expected_outputs: list[str],
) -> str:
    """Wrap a stdin/stdout program to run against each (stdin, expected stdout) pair and assert."""
    cases = ", ".join(
        f"({inp!r}, {exp.strip() if isinstance(exp, str) else exp!r})"
        for inp, exp in zip(inputs, expected_outputs)
    )
    return (
        "import sys\n"
        "from io import StringIO\n"
        f"_SRC = {program!r}\n\n"
        "def _run(stdin):\n"
        "    old_in, old_out = sys.stdin, sys.stdout\n"
        "    sys.stdin = StringIO(stdin)\n"
        "    cap = StringIO()\n"
        "    sys.stdout = cap\n"
        "    try:\n"
        '        exec(compile(_SRC, "<program>", "exec"), {"__name__": "__main__"})\n'
        "    finally:\n"
        "        sys.stdin, sys.stdout = old_in, old_out\n"
        "    return cap.getvalue().strip()\n\n"
        f"_cases = [{cases}]\n"
        "for _i, (_in, _exp) in enumerate(_cases):\n"
        "    _out = _run(_in)\n"
        "    assert _out == _exp, f'test {_i + 1}: expected {_exp!r}, got {_out!r}'\n"
    )


def run_program(
    program: str,
    stdin: str,
    python_executable: str | Path = BUILD_PYTHON,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Run a stdin/stdout program on one input and capture its stdout.

    Returns a (success, captured stdout or an error message) tuple.
    """
    ok, output = execute(
        code=_capture_harness(program, stdin),
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    return ok, (output.strip() if ok else output)


def evaluate_record(
    program: str,
    inputs: list[str],
    expected_outputs: list[str],
    python_executable: str | Path = BUILD_PYTHON,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Check a stdin/stdout program against a problem's (inputs, expected_outputs).

    Returns a (passed, empty string or the failure message) tuple.
    """
    return execute(
        code=_test_harness(program, inputs, expected_outputs),
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )


def _pip(
    python_executable: str | Path,
    *args: str,
) -> bool:
    """Run pip inside a venv; return whether it succeeded."""
    result = subprocess.run(
        [str(python_executable), "-m", "pip", "--quiet", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _normalise(
    name: str,
) -> str:
    """Normalise a distribution name for comparison (lowercase, unify - and _).

    Returns the normalised name.
    """
    return name.strip().lower().replace("_", "-")


def _installed_distributions(
    python_executable: str | Path,
) -> dict[str, str]:
    """The packages currently installed in a venv, as {normalised name: version}.

    Returns the distribution map (empty on failure).
    """
    result = subprocess.run(
        [str(python_executable), "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
    )
    dists: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "==" in line:
            name, _, version = line.partition("==")
            dists[_normalise(name)] = version.strip()
    return dists


class InstallResult(NamedTuple):
    """Outcome of a temporary_packages install: what failed, and the versions of what installed."""

    failed: set[str]
    versions: dict[str, str]


@contextmanager
def temporary_packages(
    python_executable: str | Path,
    packages: list[str],
) -> Iterator[InstallResult]:
    """Install `packages` into a venv for the duration of the block, then FULLY uninstall them.

    Everything the install added (the named packages AND their transitive dependencies) is removed
    on exit, so the venv returns to its pre-install state - important for a clean, reproducible set
    of gold solutions where each problem is evaluated in the same baseline environment.

    Yields an InstallResult with the packages that failed to install (so the caller can skip a
    problem whose libraries are unavailable) and the {name: version} of everything installed.
    """
    baseline = _installed_distributions(python_executable)
    failed: list[str] = []
    for package in sorted(set(packages)):
        if not _pip(python_executable, "install", package):
            failed.append(package)

    current = _installed_distributions(python_executable)
    # the installed version of each requested package (robust even if it was already present)
    requested = {
        _normalise(package): current[_normalise(package)]
        for package in packages
        if _normalise(package) in current
    }
    # everything the install newly added (the packages plus their dependencies), to uninstall
    added = [name for name in current if name not in baseline]
    try:
        yield InstallResult(failed=set(failed), versions=requested)
    finally:
        if added:
            _pip(python_executable, "uninstall", "-y", *added)


class Environment:
    """A cached virtual environment the harness installs response libraries into, then runs with.

    Packages already installed in this environment are tracked so repeated answers needing the
    same library don't reinstall. Installing arbitrary packages from model output is inherent to
    running a library-agnostic benchmark - use a disposable cache directory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._installed: set[str] = set()
        if not self.python.exists():
            venv.EnvBuilder(with_pip=True, clear=False).create(self.cache_dir)

    @property
    def python(self) -> Path:
        """The environment's python interpreter.

        Returns the interpreter path.
        """
        bindir = "Scripts" if platform.system() == "Windows" else "bin"
        return self.cache_dir / bindir / "python"

    def ensure_packages(
        self,
        packages: set[str],
    ) -> set[str]:
        """Install any of `packages` not already present; return the ones that failed to install.

        Tries a single batch install, then falls back to per-package installs to pin down which
        names are bad (e.g. a hallucinated package) without failing the whole set.

        Returns the set of packages that could not be installed.
        """
        todo = sorted(packages - self._installed)
        if not todo:
            return set()

        if self._pip_install(todo):
            self._installed.update(todo)
            return set()

        # batch failed - install one at a time to isolate the bad names
        failed: set[str] = set()
        for package in todo:
            if self._pip_install([package]):
                self._installed.add(package)
            else:
                failed.add(package)
        return failed

    def _pip_install(
        self,
        packages: list[str],
    ) -> bool:
        """Run pip install for `packages` in this environment; return whether it succeeded."""
        result = subprocess.run(
            [str(self.python), "-m", "pip", "install", "--quiet", *packages],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def evaluate(
        self,
        program: str,
        inputs: list[str],
        expected_outputs: list[str],
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> tuple[bool, str]:
        """Run a program against a problem's tests using this environment's interpreter.

        Returns a (passed, empty string or the failure message) tuple.
        """
        return evaluate_record(
            program=program,
            inputs=inputs,
            expected_outputs=expected_outputs,
            python_executable=self.python,
            timeout_seconds=timeout_seconds,
        )
