"""Tests for reward computation and the GRPO reward function factory."""

import pytest

from src.grpo.rewards import (
    compute_reward,
    load_reward_profile,
    make_library_reuse_reward,
)
from src.response.parser import ParsedResponse
from src.response.pipeline import ProcessedResponse


def _dummy_parsed(
    reasoning: str = "thinking",
    answer: str | None = None,
    code: str = "print(1)",
    valid: bool = True,
    std_libs: list[str] | None = None,
    ext_libs: list[str] | None = None,
    lib_usage: dict[str, list[dict]] | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse for reward tests, with sensible defaults."""
    return ParsedResponse(
        reasoning=reasoning,
        answer=answer if answer is not None else code,
        code=code,
        valid=valid,
        error=None,
        std_libs=std_libs or [],
        ext_libs=ext_libs or [],
        lib_usage=lib_usage or {},
        logical_lines=1,
        maintainability_index=100.0,
    )


class TestComputeRewardComponents:
    """Tests for the individual reward components inside compute_reward."""

    def test_correct_format_requires_reasoning_answer_and_code(self):
        """Correct format needs non-empty reasoning, answer, and code - not validity."""
        weights = {"correct_format": 1.0}

        assert compute_reward(_dummy_parsed(), False, weights) == 1.0
        assert compute_reward(_dummy_parsed(reasoning=""), False, weights) == 0.0
        assert compute_reward(_dummy_parsed(answer=""), False, weights) == 0.0
        assert compute_reward(_dummy_parsed(code=""), False, weights) == 0.0
        # invalid code doesn't affect format, since format is about response shape
        assert compute_reward(_dummy_parsed(valid=False), False, weights) == 1.0

    def test_passes_syntax_check_only_cares_about_validity(self):
        """Syntax check is true whenever valid, regardless of format/emptiness."""
        weights = {"passes_syntax_check": 1.0}

        assert compute_reward(_dummy_parsed(), False, weights) == 1.0
        assert compute_reward(_dummy_parsed(code=""), False, weights) == 1.0
        assert compute_reward(_dummy_parsed(reasoning=""), False, weights) == 1.0
        assert compute_reward(_dummy_parsed(valid=False), False, weights) == 0.0

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

    def test_internal_lib_imported_and_used(self):
        """A stdlib import with recorded usage sets both imported and used."""
        parsed = _dummy_parsed(
            std_libs=["math"], lib_usage={"math": [{"call": "sqrt"}]}
        )
        weights = {"internal_lib_imported": 1.0, "internal_lib_used": 1.0}

        assert compute_reward(parsed, False, weights) == 2.0

    def test_external_lib_imported_and_used(self):
        """A third-party import with recorded usage sets both imported and used."""
        parsed = _dummy_parsed(
            ext_libs=["numpy"], lib_usage={"numpy": [{"call": "array"}]}
        )
        weights = {"external_lib_imported": 1.0, "external_lib_used": 1.0}

        assert compute_reward(parsed, False, weights) == 2.0


class TestLoadRewardProfile:
    """Tests for load_reward_profile."""

    def test_loads_a_named_profile(self, tmp_path):
        """A profile's weights are returned as a plain dict."""
        config_file = tmp_path / "rewards.yaml"
        config_file.write_text("full:\n  passed_tests: 1.0\n  correct_format: 0.1\n")

        weights = load_reward_profile(profile="full", config_file=str(config_file))

        assert weights == {"passed_tests": 1.0, "correct_format": 0.1}

    def test_unknown_component_raises(self, tmp_path):
        """A typo'd component name fails fast instead of silently never counting."""
        config_file = tmp_path / "rewards.yaml"
        config_file.write_text("full:\n  not_a_real_component: 1.0\n")

        with pytest.raises(ValueError):
            load_reward_profile(profile="full", config_file=str(config_file))


class TestComputeReward:
    """Tests for compute_reward."""

    def test_passed_gets_format_and_passed_weights(self):
        """A passing, well-formatted response gets the sum of those two weights."""
        reward = compute_reward(
            parsed=_dummy_parsed(),
            passed=True,
            weights={"correct_format": 0.1, "passed_tests": 1.0},
        )

        assert reward == 1.1

    def test_failed_gets_no_reward_for_passed_component(self):
        """A response whose code failed evaluation doesn't get the passed-tests weight."""
        reward = compute_reward(
            parsed=_dummy_parsed(),
            passed=False,
            weights={"correct_format": 0.1, "passed_tests": 1.0},
        )

        assert reward == 0.1

    def test_components_not_in_weights_dont_contribute(self):
        """A component left out of the profile never contributes, even if satisfied."""
        reward = compute_reward(
            parsed=_dummy_parsed(),
            passed=True,
            weights={"passed_tests": 1.0},
        )

        assert reward == 1.0


class TestMakeLibraryReuseReward:
    """Tests for make_library_reuse_reward."""

    def test_rebuilds_record_from_kwargs_and_scores_each_completion(self, monkeypatch):
        """Each completion is paired with its own record, zipped back from the kwargs columns."""
        seen_calls = []

        def fake_process_response(response, record):
            seen_calls.append((response, record))
            return ProcessedResponse(**vars(_dummy_parsed()), passed=True, output="")

        monkeypatch.setattr("src.grpo.rewards.process_response", fake_process_response)

        completions = [
            [{"content": "aa"}],
            [{"content": "bbb"}],
        ]

        library_reuse_reward = make_library_reuse_reward({"passed_tests": 1.0})
        rewards = library_reuse_reward(
            completions,
            inputs=[["1\n"], ["2\n"]],
            expected_outputs=[["1\n"], ["2\n"]],
        )

        assert rewards == [1.0, 1.0]
        assert seen_calls == [
            ("aa", {"inputs": ["1\n"], "expected_outputs": ["1\n"]}),
            ("bbb", {"inputs": ["2\n"], "expected_outputs": ["2\n"]}),
        ]

    def test_binds_the_given_weights(self):
        """The factory's weights argument restricts which components count."""
        library_reuse_reward = make_library_reuse_reward({"correct_format": 0.1})

        rewards = library_reuse_reward(
            [[{"content": "reasoning</think>\n```python\nprint(1)\n```"}]],
            inputs=[["irrelevant\n"]],
            expected_outputs=[["wrong\n"]],
        )

        # the code fails evaluation, so only correct_format (which is in weights)
        # contributes - passed_tests isn't even in weights, so it's irrelevant
        assert rewards == [0.1]
