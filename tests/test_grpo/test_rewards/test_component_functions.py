"""Tests for splitting a reward profile into one logged function per component."""

import pytest

from src.grpo.rewards import (
    _REWARD_CLASSES,
    _SUPPORTED_COMPONENTS,
    build_reward_functions,
)
from src.grpo.rewards.library_usage import LibraryUsageReward


# the training profile for the library-usage task (see config/rewards.yaml)
_WEIGHTS = {
    "correct_format": 0.25,
    "passes_syntax_check": 0.25,
    "external_lib_imported": 0.5,
    "external_lib_used": 1.0,
    "target_lib_imported": 0.0,
    "target_lib_used": 0.0,
}


def _completion(code: str) -> str:
    """Wrap python code as a completion with reasoning and a fenced block."""
    return f"reasoning</think>\n```python\n{code}\n```"


class TestBuildRewardFunctions:
    """Tests for build_reward_functions."""

    def test_one_named_function_per_weighted_component(self):
        """Every component in the profile becomes its own named function and weight."""
        functions, weights = build_reward_functions("library_usage", _WEIGHTS)

        # trl logs each function under its __name__, so the names are the component names
        assert [function.__name__ for function in functions] == sorted(_WEIGHTS)
        assert weights == [_WEIGHTS[name] for name in sorted(_WEIGHTS)]

    def test_each_function_returns_the_components_rate(self):
        """A function returns 1/0 per completion, so its mean is that component's rate."""
        functions, _ = build_reward_functions("library_usage", _WEIGHTS)
        by_name = {function.__name__: function for function in functions}

        completions = [
            [{"content": _completion("import requests\nrequests.get('http://x')")}],
            [{"content": _completion("import math\nprint(math.sqrt(4))")}],
        ]
        columns = {"local_modules": [[], []], "target_libs": [["requests"], []]}

        assert by_name["external_lib_used"](completions, **columns) == [1.0, 0.0]
        assert by_name["target_lib_used"](completions, **columns) == [1.0, 0.0]

    def test_weighted_sum_matches_scoring_the_whole_profile(self):
        """trl sums weight x function, which must equal the single reward's score."""
        functions, weights = build_reward_functions("library_usage", _WEIGHTS)
        completions = [
            [{"content": _completion("import requests\nrequests.get('http://x')")}]
        ]
        columns = {"local_modules": [[]], "target_libs": [["requests"]]}

        total = sum(
            weight * function(completions, **columns)[0]
            for function, weight in zip(functions, weights)
        )

        assert total == pytest.approx(0.25 + 0.25 + 0.5 + 1.0)

    def test_a_batch_is_judged_once_for_all_components(self, monkeypatch):
        """Judging is shared, so the tested reward never re-runs the generated code."""
        calls = []

        def counted_components(self, content, row):
            calls.append(content)
            return {name: True for name in _WEIGHTS}

        monkeypatch.setattr(LibraryUsageReward, "components", counted_components)

        functions, _ = build_reward_functions("library_usage", _WEIGHTS)
        completions = [[{"content": "one"}], [{"content": "two"}]]
        for function in functions:
            function(completions, local_modules=[[], []])

        # one judgement per completion, not one per completion per component
        assert calls == ["one", "two"]


def test_every_reward_reports_exactly_its_supported_components(monkeypatch):
    """The components a reward reports must match what profiles may weight it on.

    build_reward validates profiles against _SUPPORTED_COMPONENTS, and the component
    functions then look each one up by name - so the two must not drift apart.
    """
    monkeypatch.setattr(
        "src.grpo.rewards.tested_implementation.process_response",
        lambda response, record, stop_reason="unknown": _passed_response(),
    )
    row = {
        "local_modules": [],
        "target_libs": [],
        "allowlist": [],
        "aliases": {},
        "diy_terms": [],
        "inputs": ["1\n"],
        "expected_outputs": ["1\n"],
    }

    for kind, reward_class in _REWARD_CLASSES.items():
        components = reward_class({}).components(_completion("import requests"), row)

        assert set(components) == _SUPPORTED_COMPONENTS[kind], kind


def _passed_response():
    """Build a passing ProcessedResponse, so the tested reward needs no execution."""
    from lure import parse_response

    from src.response.pipeline import ProcessedResponse

    parsed = parse_response(_completion("import requests"))
    return ProcessedResponse(parsed=parsed, passed=True, output="")
