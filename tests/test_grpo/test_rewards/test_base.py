"""Tests for the shared reward base classes, helpers, and profile loading."""

import pytest

from src.grpo.rewards.base import (
    CodeReward,
    LibraryReward,
    load_reward_profile,
    rows_from_columns,
)
from src.response.parser import ParsedResponse


def _parsed(
    reasoning: str = "thinking",
    answer: str | None = "answer",
    code: str | None = "print(1)",
    valid: bool | None = True,
    std_libs: list[str] | None = None,
    ext_libs: list[str] | None = None,
    lib_usage: dict[str, list[dict]] | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse for the base-class tests."""
    return ParsedResponse(
        reasoning=reasoning,
        answer=answer,
        code=code,
        valid=valid,
        error=None,
        std_libs=std_libs or [],
        ext_libs=ext_libs or [],
        lib_usage=lib_usage or {},
        logical_lines=1,
        maintainability_index=100.0,
    )


class TestRowsFromColumns:
    """Tests for rows_from_columns."""

    def test_transposes_columns_into_rows(self):
        """Per-column lists become one dict per row, keyed by column name."""
        rows = rows_from_columns({"a": [1, 2], "b": ["x", "y"]}, n=2)

        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_skips_columns_of_the_wrong_length(self):
        """A column that isn't a list of length n is ignored, not fatal."""
        rows = rows_from_columns({"a": [1, 2], "bad": "notalist", "short": [1]}, n=2)

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


class TestCodeReward:
    """Tests for the shared CodeReward helpers."""

    def test_external_libraries_subtracts_local_modules(self):
        """The project's own modules are not counted as external libraries."""
        parsed = _parsed(ext_libs=["requests", "app"])

        assert CodeReward.external_libraries(parsed, ["app"]) == ["requests"]
        assert CodeReward.external_libraries(parsed, None) == ["requests", "app"]

    def test_score_format_and_syntax_components(self):
        """correct_format needs full shape; passes_syntax_check needs valid code."""
        reward = CodeReward({"correct_format": 0.1, "passes_syntax_check": 0.2})

        assert reward.score_format_and_syntax(0.0, _parsed()) == pytest.approx(0.3)
        # missing code -> not correctly formatted, but still counts as invalid syntax
        assert reward.score_format_and_syntax(
            0.0, _parsed(code=None, valid=None)
        ) == pytest.approx(0.0)


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
