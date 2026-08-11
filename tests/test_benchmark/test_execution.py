"""Tests for the stdin/stdout execution used by the build pipeline (bcbreuse.environment)."""

import sys

from bcbreuse.environment import evaluate_record, run_program, temporary_packages


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
