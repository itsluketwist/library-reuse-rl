"""Tests for the tested-implementation reward (execution-verified)."""

from lure import CodeBlock, analyse_code, parse_response

from src.grpo.rewards import build_reward
from src.grpo.rewards.tested_implementation import (
    TestedImplementationReward,
    execution_components,
)
from src.response.pipeline import ProcessedResponse


def _processed(
    response: str,
    passed: bool = True,
) -> ProcessedResponse:
    """Build a ProcessedResponse from a real response, as if its code had been run."""
    return ProcessedResponse(
        parsed=parse_response(response),
        passed=passed,
        output="",
    )


def _block(code: str) -> CodeBlock:
    """Analyse a code block, as the one run against the tests."""
    return analyse_code(code)


class TestExecutionComponents:
    """Tests for the execution-specific reward components.

    correct_format and passes_syntax_check are shared components, tested against
    LibraryReward/CodeReward directly in test_base.py rather than here.
    """

    def test_passed_tests_reflects_the_passed_argument(self):
        """Passed-tests just reports the caller's evaluation outcome."""
        block = _block("print(1)")

        assert execution_components(block, passed=True)["passed_tests"]
        assert not execution_components(block, passed=False)["passed_tests"]

    def test_os_import_alone_is_not_an_internal_lib(self):
        """Importing only 'os' (stdin/stdout boilerplate) doesn't count as internal."""
        components = execution_components(
            _block("import os\nos.read(0, 10)"),
            passed=False,
        )

        assert not components["internal_lib_imported"]
        assert not components["internal_lib_used"]

    def test_internal_lib_imported_but_unused(self):
        """A stdlib import that is never referenced is imported but not used."""
        components = execution_components(_block("import math"), passed=False)

        assert components["internal_lib_imported"]
        assert not components["internal_lib_used"]

    def test_external_lib_imported_and_used(self):
        """A referenced third-party import is both imported and used."""
        components = execution_components(
            _block("import numpy as np\nnp.array([1])"),
            passed=False,
        )

        assert components["external_lib_imported"]
        assert components["external_lib_used"]

    def test_no_code(self):
        """Without code, only passed_tests can be reported (and is False in practice)."""
        components = execution_components(None, passed=False)

        assert not any(components.values())


class TestTestedImplementationReward:
    """Tests for the class wrapper that executes code and scores it."""

    def test_runs_code_against_the_rows_tests_and_scores(self, monkeypatch):
        """score runs process_response with the row's tests, then scores it."""
        seen = []

        def fake_process_response(response, record, stop_reason="unknown"):
            seen.append((response, record))
            return _processed("```python\nprint(1)\n```", passed=True)

        monkeypatch.setattr(
            "src.grpo.rewards.tested_implementation.process_response",
            fake_process_response,
        )

        reward = TestedImplementationReward({"passed_tests": 1.0})
        out = reward(
            [[{"content": "aa"}], [{"content": "bbb"}]],
            inputs=[["1\n"], ["2\n"]],
            expected_outputs=[["1\n"], ["2\n"]],
        )

        assert out == [1.0, 1.0]
        assert seen == [
            ("aa", {"inputs": ["1\n"], "expected_outputs": ["1\n"]}),
            ("bbb", {"inputs": ["2\n"], "expected_outputs": ["2\n"]}),
        ]

    def test_score_composes_shared_format_and_syntax_with_execution(self, monkeypatch):
        """score adds the shared correct_format/passes_syntax_check to execution."""
        monkeypatch.setattr(
            "src.grpo.rewards.tested_implementation.process_response",
            lambda response, record, stop_reason="unknown": _processed(
                "thinking</think>\nI would just use the requests library.",
                passed=True,
            ),
        )

        weights = {
            "correct_format": 1.0,
            "passes_syntax_check": 1.0,
            "passed_tests": 1.0,
        }
        reward = TestedImplementationReward(weights)

        score = reward.score("aa", {"inputs": ["1\n"], "expected_outputs": ["1\n"]})

        # correct_format fires without code; passes_syntax_check does not (no code)
        assert score == 1.0 + 1.0

    def test_build_reward_selects_this_class(self):
        """The factory resolves the tested-implementation kind to this reward."""
        reward = build_reward("implement_tested", {"passed_tests": 1.0})

        assert isinstance(reward, TestedImplementationReward)
