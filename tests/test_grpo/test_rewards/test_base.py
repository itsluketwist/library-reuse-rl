"""Tests for the shared reward base classes, helpers, and profile loading."""

import pytest
from lure import ParsedResponse, parse_response

from src.grpo.rewards.base import (
    CodeReward,
    LibraryReward,
    load_reward_profile,
    rows_from_columns,
)


def _parsed(
    reasoning: str = "thinking",
    # real prose, since an answer that happens to be valid python (e.g. one word) is code
    answer: str | None = "Here is my answer.",
    code: str | None = "print(1)",
    stop_reason: str = "eos",
) -> ParsedResponse:
    """Build a ParsedResponse for the base-class tests, by parsing a real response."""
    body = ""
    if answer is not None:
        body = answer
        if code is not None:
            body += f"\n```python\n{code}\n```"
    response = f"<think>{reasoning}</think>\n{body}" if reasoning else body
    return parse_response(response, stop_reason=stop_reason)


class TestRowsFromColumns:
    """Tests for rows_from_columns."""

    def test_transposes_columns_into_rows(self):
        """Per-column lists become one dict per row, keyed by column name."""
        rows = rows_from_columns({"a": [1, 2], "b": ["x", "y"]}, n=2)

        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_skips_columns_of_the_wrong_length(self):
        """A column that isn't a list of length n is ignored, not fatal."""
        # "bad" is deliberately not a list, to exercise the wrong-type handling
        columns: dict = {"a": [1, 2], "bad": "notalist", "short": [1]}
        rows = rows_from_columns(columns, n=2)

        assert rows == [{"a": 1}, {"a": 2}]


class TestLibraryRewardWiring:
    """Tests for LibraryReward's GRPO calling convention and weight bookkeeping."""

    def test_call_scores_each_completion_with_its_row(self):
        """__call__ pairs each completion with its transposed row and calls score."""

        class _Echo(LibraryReward):
            def score(self, content, row):
                # length of content plus the row's own weight column, to prove pairing
                return len(content) + row["bonus"]

        reward = _Echo({})
        out = reward(
            [[{"content": "aa"}], [{"content": "bbbb"}]],
            bonus=[10, 20],
        )

        assert out == [12, 24]

    def test_add_only_counts_active_satisfied_components(self):
        """_add contributes a weight only when it's present, non-zero, and satisfied."""
        reward = LibraryReward({"present": 0.5, "zero": 0.0})

        assert reward._add(0.0, "present", True) == 0.5
        assert reward._add(0.0, "present", False) == 0.0
        assert reward._add(0.0, "absent", True) == 0.0
        assert reward._add(0.0, "zero", True) == 0.0

    def test_stop_reason_needs_both_eos_token_id_and_completion_ids(self):
        """Truncation can only be judged when both pieces of information are present."""
        reward = LibraryReward({}, eos_token_id=2)

        assert reward._stop_reason({"completion_ids": [1, 3]}) == "length"
        assert reward._stop_reason({"completion_ids": [1, 2]}) == "eos"
        assert (
            reward._stop_reason({}) == "unknown"
        )  # no completion_ids (e.g. a direct unit test)
        assert (
            LibraryReward({})._stop_reason({"completion_ids": [1, 3]}) == "unknown"
        )  # no eos_token_id known

    def test_correct_format_needs_reasoning_and_answer_only(self):
        """correct_format needs reasoning and an answer - no code required."""
        reward = LibraryReward({"correct_format": 0.1})

        assert reward.has_correct_format(_parsed(reasoning="thinking")) is True
        assert reward.has_correct_format(_parsed(reasoning="")) is False
        assert reward.has_correct_format(_parsed(answer=None)) is False

    def test_correct_format_rejects_truncated_completions(self):
        """A truncated completion never earns correct_format, even with full shape."""
        reward = LibraryReward({"correct_format": 0.1}, eos_token_id=2)

        assert reward.has_correct_format(_parsed(stop_reason="length")) is False
        assert reward.has_correct_format(_parsed(stop_reason="eos")) is True

    def test_score_sums_the_weights_of_satisfied_components(self):
        """score weights whatever components() reports, ignoring unweighted ones."""

        class _Fixed(LibraryReward):
            def components(self, content, row):
                return {"present": True, "absent": True, "unsatisfied": False}

        reward = _Fixed({"present": 0.5, "unsatisfied": 1.0})

        assert reward.score("content", {}) == pytest.approx(0.5)


class TestCodeReward:
    """Tests for the shared CodeReward helpers."""

    def test_parse_subtracts_the_rows_local_modules(self):
        """The project's own modules (the row's local_modules) are not external."""
        reward = CodeReward({})
        content = "r</think>\n```python\nimport requests\nimport server\n```"

        parsed = reward.parse(content, {"local_modules": ["server"]})
        assert parsed.external_imports == {"requests"}
        assert reward.parse(content, {}).external_imports == {"requests", "server"}

    def test_valid_syntax_needs_code_that_parses(self):
        """passes_syntax_check needs python code, all of which parses."""
        reward = CodeReward({"passes_syntax_check": 0.2})

        assert reward.has_valid_syntax(_parsed(code="print(1)")) is True
        assert reward.has_valid_syntax(_parsed(code="def broken(:")) is False
        assert reward.has_valid_syntax(_parsed(code=None)) is False


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
