"""Run candidate stdin/stdout programs, and manage the venv the harness installs into.

Two layers. `run_program` / `evaluate_record` execute a stdin/stdout program in a sandboxed
subprocess (time + memory limited) - used both to mint oracle outputs at build time and to score
answers. `Environment` wraps a cached virtual environment the harness installs a response's
libraries into (arbitrary, dynamic) before running it, so each answer is evaluated with exactly
the third-party packages it reached for.
"""

import os
import platform
import shutil
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
    resource = None  # ty: ignore[invalid-assignment]


# generous enough for heavy scientific libraries (sklearn/scipy/statsmodels import slowly), so a
# genuinely dual-solvable task is not dropped just for import latency under a tight cap
_DEFAULT_TIMEOUT_SECONDS = 30.0
# a cap on the address space the child may reserve, which is far more than the memory it really
# uses: importing a scientific library alone reserves hundreds of megabytes. so the cap is set
# well above what a correct solution needs, and only stops a runaway one
_DEFAULT_MEMORY_MB = 4096

# numeric libraries start one worker thread per cpu core by default, and each thread reserves
# its own memory. on a many-core cluster node that alone went over the memory cap, so correct
# library solutions failed with "OpenBLAS error: Memory allocation still failed". one thread is
# plenty for benchmark-sized inputs, and makes a run behave the same on a laptop and a cluster
_SINGLE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

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

    Numeric libraries are held to a single thread (see _SINGLE_THREAD_ENV).

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
                env={**os.environ, **_SINGLE_THREAD_ENV},
            )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"

    if result.returncode == 0:
        return True, result.stdout
    error = result.stderr.strip() or result.stdout.strip()
    return False, error or f"exit code {result.returncode}"


def warm_up_imports(
    import_names: set[str],
    python_executable: str | Path = BUILD_PYTHON,
    timeout_seconds: float = 300.0,
) -> None:
    """Import each module once, so that the runs that follow don't pay for a first import.

    The first import of a large library in a new environment can take most of a test's time
    limit: its files are read and checked for the first time. Without this, that cost falls on
    whichever solution happens to use the library first, and can fail it. A module that can't be
    imported is skipped: the solution that needs it will report that itself.
    """
    if not import_names:
        return
    code = "\n".join(
        f"try:\n    import {name}\nexcept Exception:\n    pass"
        for name in sorted(import_names)
    )
    execute(
        code=code,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )


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
    """A virtual environment the harness installs response libraries into, then runs with.

    By default each Environment gets its own uniquely named venv and deletes it on close, so
    evaluations running at the same time can never share - and corrupt - one directory. Name a
    `cache_dir` to reuse a venv across runs instead: packages already installed there are
    tracked and not reinstalled, at the cost of that directory being shared with anything else
    pointed at it. A named directory is always kept; only one this class created is deleted.

    `scratch_root` places the disposable venv somewhere specific (it defaults to the system
    temp directory), for when the default location is too small or on the wrong filesystem.

    Installing arbitrary packages from model output is inherent to running a library-agnostic
    benchmark, which is why a disposable environment is the default.

    Use it as a context manager, or call close(), so a disposable venv is cleaned up.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        scratch_root: str | Path | None = None,
    ) -> None:
        # only a venv this class created is ours to delete; a named one belongs to the caller
        self._owned = cache_dir is None
        if cache_dir is None:
            if scratch_root is not None:
                Path(scratch_root).mkdir(parents=True, exist_ok=True)
            cache_dir = tempfile.mkdtemp(prefix="libreusebench-env-", dir=scratch_root)

        # resolved to absolute: execute() runs the interpreter with a different subprocess cwd
        # per call, and a relative path would then resolve against that cwd instead of this one
        self.cache_dir = Path(cache_dir).resolve()
        self._installed: set[str] = set()
        if not self.python.exists():
            venv.EnvBuilder(with_pip=True, clear=False).create(self.cache_dir)

        # fail here rather than letting every subsequent execute() record a wrong answer,
        # which would look like a model scoring zero instead of a broken environment
        if not self.python.exists():
            raise RuntimeError(
                f"no python interpreter at {self.python} after building the environment"
            )

    def close(self) -> None:
        """Delete the environment, unless the caller named it (see __init__)."""
        if self._owned:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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

    def warm_up(
        self,
        import_names: set[str],
    ) -> None:
        """Import each module once in this environment, before anything is timed (see warm_up_imports)."""
        warm_up_imports(import_names, python_executable=self.python)

    def installed_distributions(self) -> dict[str, str]:
        """Inspect every distribution installed in the evaluation environment.

        This includes transitively installed dependencies, which makes the recorded
        environment useful when debugging more than just the requested top-level packages.

        Returns the {normalised distribution name: version} map.
        """
        return _installed_distributions(self.python)

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

    def evaluate_function(
        self,
        code: str,
        function_name: str,
        test_inputs: list[str],
        expected_outputs: list[str],
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> tuple[bool, str]:
        """Call a solution's function on a function task's tests using this environment's interpreter.

        Returns a (passed, empty string or the failure message) tuple.
        """
        # imported here, not at the top of the file, because functions.py imports this module
        from libreusebench.functions import evaluate_function_record

        return evaluate_function_record(
            code=code,
            function_name=function_name,
            test_inputs=test_inputs,
            expected_outputs=expected_outputs,
            python_executable=self.python,
            timeout_seconds=timeout_seconds,
        )
