"""Tests for the library-usage reward (external reuse, no execution)."""

import pytest

from src.grpo.rewards import build_reward, load_reward_profile
from src.grpo.rewards.library_usage import LibraryUsageReward


# the external-reuse components on, so each test can read off which fired
_WEIGHTS = {
    "external_lib_imported": 1.0,
    "external_lib_used": 1.0,
    "target_lib_used": 1.0,
}


def _completion(code: str) -> str:
    """Wrap python code as a completion with reasoning and a fenced block."""
    return f"reasoning</think>\n```python\n{code}\n```"


class TestLibraryUsageReward:
    """Tests for LibraryUsageReward.score."""

    def test_used_target_library_earns_import_use_and_target(self):
        """A used third-party import from the target list fires all three components."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nrequests.get('http://x')"),
            {"local_modules": [], "target_libs": ["requests"]},
        )

        assert score == 3.0

    def test_external_library_not_in_target_list_skips_the_target_credit(self):
        """An external lib that isn't a target one earns import+use only."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nrequests.get('http://x')"),
            {"local_modules": [], "target_libs": ["httpx"]},
        )

        assert score == 2.0

    def test_stdlib_only_earns_nothing(self):
        """Reaching for the standard library is not external reuse here."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import math\nprint(math.sqrt(4))"),
            {"local_modules": [], "target_libs": []},
        )

        assert score == 0.0

    def test_project_local_import_is_not_external(self):
        """Importing the project's own modules doesn't count as library reuse."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("from app.models import User\nUser()"),
            {"local_modules": ["app"], "target_libs": []},
        )

        assert score == 0.0

    def test_relative_import_is_not_external(self):
        """Relative imports are the project's own code, never external reuse."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("from . import helpers\nhelpers.run()"),
            {"local_modules": [], "target_libs": []},
        )

        assert score == 0.0

    def test_imported_but_unused_external_lib_earns_import_only(self):
        """An external import that is never called earns import but not use/target."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nprint('hello')"),
            {"local_modules": [], "target_libs": ["requests"]},
        )

        assert score == 1.0

    def test_aggregates_library_use_across_every_code_block(self):
        """A library in a shorter block counts even when another block is longest."""
        reward = LibraryUsageReward(_WEIGHTS)
        content = """reasoning</think>
```python
import requests
requests.get("https://example.com")
```
```python
def longer_hand_written_example(value):
    result = value.strip()
    return result.lower()
```
"""

        score = reward.score(
            content,
            {"local_modules": [], "target_libs": ["requests"]},
        )

        assert score == 3.0

    def test_correct_format_does_not_require_code(self):
        """correct_format is shared with the other rewards - no code block needed."""
        reward = LibraryUsageReward({"correct_format": 1.0})

        score = reward.score(
            "reasoning</think>\nJust use the requests library for this.",
            {"local_modules": [], "target_libs": []},
        )

        assert score == 1.0

    def test_correct_format_rejects_a_truncated_completion(self):
        """A truncated completion never earns correct_format, even with full shape."""
        weights = {"correct_format": 1.0}
        reward = LibraryUsageReward(weights, eos_token_id=2)

        row = {"local_modules": [], "target_libs": [], "completion_ids": [1, 3]}
        score = reward.score(
            _completion("import requests\nrequests.get('http://x')"),
            row,
        )

        assert score == 0.0

    def test_target_import_needs_no_use(self):
        """target_lib_imported mirrors external_lib_imported, for target libraries only."""
        reward = LibraryUsageReward({"target_lib_imported": 1.0})

        unused_target = reward.score(
            _completion("import requests\nprint('hello')"),
            {"local_modules": [], "target_libs": ["requests"]},
        )
        other_library = reward.score(
            _completion("import httpx\nhttpx.get('http://x')"),
            {"local_modules": [], "target_libs": ["requests"]},
        )

        assert (unused_target, other_library) == (1.0, 0.0)

    def test_build_reward_selects_this_class(self):
        """The factory resolves the library-usage kind to this reward."""
        reward = build_reward("library_usage", _WEIGHTS)

        assert isinstance(reward, LibraryUsageReward)

    def test_build_reward_rejects_incompatible_profile(self):
        """A tested-code component cannot silently become a no-op for this task."""
        with pytest.raises(ValueError, match="does not score"):
            build_reward("library_usage", {"passed_tests": 1.0})


# one response of each kind, scored by experiment 01's three main profiles in
# config/rewards.yaml. "numpy" is the task's only target library
_PROFILE_CASES = {
    "hand-written": "print(sum([1, 2]))",
    "unused target import": "import numpy\nprint(sum([1, 2]))",
    "uses another library": "import pandas as pd\npd.Series([1, 2]).sum()",
    "uses a target library": "import numpy as np\nnp.sum([1, 2])",
}


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "hand-written",
            {
                "lu-import": 1.0,
                "lu-import-matched": 0.5,
                "lu-usage": 0.5,
                "lu-target": 0.5,
            },
        ),
        (
            "unused target import",
            {
                "lu-import": 2.0,
                "lu-import-matched": 2.0,
                "lu-usage": 1.0,
                "lu-target": 1.0,
            },
        ),
        (
            "uses another library",
            {
                "lu-import": 2.0,
                "lu-import-matched": 2.0,
                "lu-usage": 2.0,
                "lu-target": 0.5,
            },
        ),
        (
            "uses a target library",
            {
                "lu-import": 2.0,
                "lu-import-matched": 2.0,
                "lu-usage": 2.0,
                "lu-target": 2.0,
            },
        ),
    ],
)
def test_main_profiles_score_each_kind_of_response(
    case: str,
    expected: dict[str, float],
):
    """The import, usage and target rewards differ only in which library use they pay."""
    row = {"local_modules": [], "target_libs": ["numpy"]}

    scores = {
        profile: LibraryUsageReward(load_reward_profile(profile)).score(
            _completion(_PROFILE_CASES[case]),
            row,
        )
        for profile in expected
    }

    assert scores == pytest.approx(expected)


@pytest.mark.parametrize("case", list(_PROFILE_CASES))
def test_half_target_profile_sits_between_usage_and_target(case: str):
    """lu-target-half pays every kind of response the average of lu-usage and lu-target."""
    row = {"local_modules": [], "target_libs": ["numpy"]}

    scores = {
        profile: LibraryUsageReward(load_reward_profile(profile)).score(
            _completion(_PROFILE_CASES[case]),
            row,
        )
        for profile in ["lu-usage", "lu-target", "lu-target-half"]
    }

    assert scores["lu-target-half"] == pytest.approx(
        (scores["lu-usage"] + scores["lu-target"]) / 2
    )
