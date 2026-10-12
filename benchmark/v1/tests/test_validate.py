"""Tests for the stage-05 library-solution prompt in benchmark/pipeline/05_validate.py.

The stage is a numbered script (not importable by name), so it is loaded by path. Only the pure
prompt-building logic is exercised here - no model is called.
"""

import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[2] / "benchmark" / "pipeline" / "05_validate.py"
_spec = importlib.util.spec_from_file_location("validate_stage", _PATH)
assert _spec is not None and _spec.loader is not None
validate_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_stage)


class TestLibraryPrompt:
    """The library-solution prompt demands a third-party library and stdin/stdout I/O."""

    def test_demands_library_and_stdio(self) -> None:
        prompt = validate_stage.build_lib_prompt("Sort the numbers on stdin.")
        assert "third-party" in prompt
        assert "standard input" in prompt and "standard output" in prompt
        assert "Sort the numbers on stdin." in prompt


class TestClassifyFailure:
    """Failures are bucketed so install/logic/crash/timeout can be told apart on the record."""

    def test_install_failure(self) -> None:
        assert (
            validate_stage.classify_failure("install failed: ['nope']")
            == "install failed"
        )

    def test_timeout(self) -> None:
        assert validate_stage.classify_failure("timeout after 10s") == "timeout"

    def test_output_mismatch_from_assertion(self) -> None:
        # the test harness asserts on stdout, so an AssertionError is a genuine disagreement
        message = (
            "Traceback (most recent call last):\n"
            "AssertionError: test 3: expected '1.5', got '1.50'"
        )
        assert validate_stage.classify_failure(message) == "output mismatch"

    def test_runtime_error_for_other_exceptions(self) -> None:
        message = "Traceback (most recent call last):\nKeyError: 'value'"
        assert validate_stage.classify_failure(message) == "runtime error"

    def test_detail_keeps_the_informative_last_line(self) -> None:
        message = "Traceback (most recent call last):\nValueError: bad input"
        assert validate_stage.failure_detail(message) == "ValueError: bad input"


class TestRetryEnvironmentFailure:
    """A prior environment failure (install/timeout) is worth retrying as-is; a behavioural one isn't."""

    def test_none_for_behavioural_failure(self) -> None:
        prior = {"drop_reason": "output mismatch", "gold_extlib": "print(1)"}
        assert validate_stage.retry_environment_failure({}, prior) is None

    def test_none_without_a_stored_solution(self) -> None:
        prior = {"drop_reason": "install failed"}
        assert validate_stage.retry_environment_failure({}, prior) is None

    def test_none_for_a_fresh_seed_with_no_prior_attempt(self) -> None:
        # main() only calls this when a prior record exists, but the gate itself should still
        # fail safe if given an empty prior
        assert validate_stage.retry_environment_failure({}, {}) is None
