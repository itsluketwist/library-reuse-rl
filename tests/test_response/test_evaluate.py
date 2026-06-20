"""Tests for stdin/stdout sandboxed code evaluation."""

import sys

from src.response.evaluate import evaluate_record


# tests run against the current interpreter rather than the .venv-eval sandbox,
# so they don't depend on `setup/setup_eval_venv.sh` having been run
PYTHON = sys.executable


class TestEvaluateRecord:
    """Tests for evaluate_record."""

    def test_simple_echo(self):
        """A program that echoes a line of input passes a single test case."""
        code = "line = input()\nprint(line)"
        record = {
            "inputs": ["hello\n", "world\n"],
            "expected_outputs": ["hello\n", "world\n"],
        }

        passed, _ = evaluate_record(code=code, record=record, python_executable=PYTHON)

        assert passed is True

    def test_sum_two_numbers(self):
        """A program that sums two stdin-provided numbers passes."""
        code = "a = int(input())\nb = int(input())\nprint(a + b)"
        record = {"inputs": ["2\n3\n", "10\n20\n"], "expected_outputs": ["5\n", "30\n"]}

        passed, _ = evaluate_record(code=code, record=record, python_executable=PYTHON)

        assert passed is True

    def test_wrong_output_is_caught(self):
        """A program that prints the wrong answer fails evaluation."""
        code = "print('wrong')"
        record = {"inputs": ["test\n"], "expected_outputs": ["test\n"]}

        passed, output = evaluate_record(
            code=code, record=record, python_executable=PYTHON
        )

        assert passed is False
        assert "wrong" in output

    def test_timeout_is_caught(self):
        """A program that hangs is killed and reported as a timeout, not a crash."""
        code = "while True:\n    pass"
        record = {"inputs": ["x\n"], "expected_outputs": ["x\n"]}

        passed, output = evaluate_record(
            code=code,
            record=record,
            python_executable=PYTHON,
            timeout_seconds=0.5,
        )

        assert passed is False
        assert "timeout" in output

    def test_missing_eval_python_raises_clear_error(self):
        """A missing evaluation interpreter raises a clear, actionable error."""
        try:
            evaluate_record(
                code="print('hi')",
                record={"inputs": ["\n"], "expected_outputs": ["\n"]},
                python_executable="/nonexistent/python",
            )
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert "setup_eval_venv.sh" in str(exc)
