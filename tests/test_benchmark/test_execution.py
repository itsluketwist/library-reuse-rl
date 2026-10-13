"""Tests for the stdin/stdout execution used by the build pipeline (libreusebench.environment)."""

import platform
import sys
import venv
from pathlib import Path

import pytest
from libreusebench.environment import (
    Environment,
    evaluate_record,
    run_program,
    temporary_packages,
    warm_up_imports,
)


class TestRunProgram:
    """run_program feeds stdin to a program and returns its captured stdout."""

    def test_captures_stdout(self) -> None:
        ok, out = run_program("print(int(input()) * 2)", stdin="5\n")
        assert ok
        assert out == "10"

    def test_multiline_io(self) -> None:
        program = "import sys\nfor line in sys.stdin:\n    print(line.strip()[::-1])\n"
        ok, out = run_program(program, stdin="abc\nxy\n")
        assert ok
        assert out == "cba\nyx"

    def test_error_reported(self) -> None:
        ok, out = run_program("raise ValueError('boom')", stdin="")
        assert not ok
        assert "boom" in out or "ValueError" in out


@pytest.fixture
def stub_venv(monkeypatch):
    """Replace venv building with a marker file, so these tests don't pay for a real venv."""

    def _create(self, path) -> None:
        interpreter = Path(path) / (
            "Scripts" if platform.system() == "Windows" else "bin"
        )
        interpreter.mkdir(parents=True, exist_ok=True)
        (interpreter / "python").write_text("")

    monkeypatch.setattr(venv.EnvBuilder, "create", _create)


def test_numeric_libraries_are_held_to_one_thread() -> None:
    """Each thread of a numeric library reserves memory, which broke the memory cap on a cluster."""
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ok, out = run_program(
        f"import os\nprint(*(os.environ[name] for name in {names!r}))",
        stdin="",
    )
    assert ok
    assert out == "1 1 1"


def test_warming_up_skips_a_module_that_cannot_be_imported() -> None:
    """A missing module must not stop the warm-up: the solution that needs it reports it."""
    warm_up_imports({"json", "no_such_module_for_this_test"})


class TestEnvironmentLifecycle:
    """Environment is disposable by default, and reused only when the caller names it."""

    def test_disposable_by_default(self, tmp_path, stub_venv) -> None:
        """With no cache_dir the venv is built somewhere unique and deleted on close, so
        evaluations running at the same time can never share (and corrupt) one directory."""
        with Environment(scratch_root=tmp_path) as env:
            directory = env.cache_dir
            assert directory.exists()
            assert directory.parent == tmp_path.resolve()

        assert not directory.exists()

    def test_concurrent_environments_never_collide(self, tmp_path, stub_venv) -> None:
        with Environment(scratch_root=tmp_path) as first:
            with Environment(scratch_root=tmp_path) as second:
                assert first.cache_dir != second.cache_dir

    def test_named_cache_dir_is_reused_and_kept(self, tmp_path, stub_venv) -> None:
        """Naming a directory is what opts into caching - it survives close, so the next
        run reuses the packages already installed there."""
        cache_dir = tmp_path / "shared-venv"

        with Environment(cache_dir=cache_dir) as env:
            assert env.cache_dir == cache_dir.resolve()

        assert cache_dir.exists()

    def test_broken_environment_raises(self, tmp_path, monkeypatch) -> None:
        """A venv that builds without producing an interpreter raises immediately, rather
        than letting every later execute() record a wrong answer for a broken environment."""
        # stand in for a venv build that silently produces no interpreter
        monkeypatch.setattr(venv.EnvBuilder, "create", lambda self, path: None)

        with pytest.raises(RuntimeError, match="no python interpreter"):
            Environment(cache_dir=tmp_path / "broken-venv")


class TestEvaluateRecord:
    """evaluate_record checks a program against (stdin, expected) pairs."""

    _PROGRAM = "print(sum(int(x) for x in input().split()))"

    def test_passes_matching_tests(self) -> None:
        passed, _ = evaluate_record(
            self._PROGRAM, inputs=["1 2 3\n", "10 20\n"], expected_outputs=["6", "30"]
        )
        assert passed

    def test_fails_wrong_output(self) -> None:
        passed, message = evaluate_record(
            self._PROGRAM, inputs=["1 2\n"], expected_outputs=["99"]
        )
        assert not passed
        assert "expected" in message


class TestTemporaryPackages:
    """Installing no packages is a clean no-op (offline - just inspects the current venv)."""

    def test_empty_is_clean(self) -> None:
        with temporary_packages(sys.executable, []) as install:
            assert install.failed == set()
            assert install.versions == {}
