"""Tests for the tested-implementation reward (execution-verified) and compute_reward."""

from src.grpo.rewards import build_reward
from src.grpo.rewards.tested_implementation import (
    TestedImplementationReward,
    compute_reward,
)
from src.response.parser import ParsedCodeBlock, ParsedResponse
from src.response.pipeline import ProcessedResponse


def _dummy_parsed(
    reasoning: str = "thinking",
    answer: str | None = None,
    code: str = "print(1)",
    valid: bool | None = True,
    std_libs: list[str] | None = None,
    ext_libs: list[str] | None = None,
    lib_usage: dict[str, list[dict]] | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse for reward tests, with sensible defaults."""
    block = ParsedCodeBlock(
        language="python",
        text=code,
        valid=valid,
        error=None,
        std_libs=std_libs or [],
        ext_libs=ext_libs or [],
        lib_usage=lib_usage or {},
        logical_lines=1,
        maintainability_index=100.0,
        selected_for_tests=True,
    )
    return ParsedResponse(
        reasoning=reasoning,
        answer=answer if answer is not None else code,
        code_blocks=[block],
    )


class TestComputeRewardComponents:
    """Tests for the individual reward components inside compute_reward.

    correct_format and passes_syntax_check are shared components, tested against
    LibraryReward/CodeReward directly in test_base.py rather than here.
    """

    def test_passed_tests_reflects_the_passed_argument(self):
        """Passed-tests just reports the caller's evaluation outcome."""
        weights = {"passed_tests": 1.0}

        assert compute_reward(_dummy_parsed(), True, weights) == 1.0
        assert compute_reward(_dummy_parsed(), False, weights) == 0.0

    def test_os_import_alone_is_not_an_internal_lib(self):
        """Importing only 'os' (stdin/stdout boilerplate) doesn't count as internal."""
        parsed = _dummy_parsed(std_libs=["os"], lib_usage={"os": [{"call": "read"}]})
        weights = {"internal_lib_imported": 1.0, "internal_lib_used": 1.0}

        assert compute_reward(parsed, False, weights) == 0.0

    def test_internal_lib_imported_but_unused(self):
        """A stdlib import with no recorded usage sets imported but not used."""
        parsed = _dummy_parsed(std_libs=["math"])

        assert compute_reward(parsed, False, {"internal_lib_imported": 1.0}) == 1.0
        assert compute_reward(parsed, False, {"internal_lib_used": 1.0}) == 0.0

    def test_external_lib_imported_and_used(self):
        """A third-party import with recorded usage sets both imported and used."""
        parsed = _dummy_parsed(
            ext_libs=["numpy"], lib_usage={"numpy": [{"call": "array"}]}
        )
        weights = {"external_lib_imported": 1.0, "external_lib_used": 1.0}

        assert compute_reward(parsed, False, weights) == 2.0


class TestTestedImplementationReward:
    """Tests for the class wrapper that executes code and scores it."""

    def test_runs_code_against_the_rows_tests_and_scores(self, monkeypatch):
        """score runs process_response with the row's tests, then compute_reward."""
        seen = []

        def fake_process_response(response, record):
            seen.append((response, record))
            return ProcessedResponse(**vars(_dummy_parsed()), passed=True, output="")

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

    def test_score_composes_shared_format_and_syntax_with_compute_reward(
        self, monkeypatch
    ):
        """score adds the shared correct_format/passes_syntax_check, then compute_reward."""
        monkeypatch.setattr(
            "src.grpo.rewards.tested_implementation.process_response",
            lambda response, record: ProcessedResponse(
                **vars(_dummy_parsed(code="", answer="just use requests", valid=None)),
                passed=True,
                output="",
            ),
        )

        weights = {
            "correct_format": 1.0,
            "passes_syntax_check": 1.0,
            "passed_tests": 1.0,
        }
        reward = TestedImplementationReward(weights)

        score = reward.score("aa", {"inputs": ["1\n"], "expected_outputs": ["1\n"]})

        # correct_format fires without code; passes_syntax_check does not (no valid block)
        assert score == 1.0 + 1.0

    def test_build_reward_selects_this_class(self):
        """The factory resolves the tested-implementation kind to this reward."""
        reward = build_reward("implement_tested", {"passed_tests": 1.0})

        assert isinstance(reward, TestedImplementationReward)
