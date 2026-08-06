"""Tests for the library-usage reward (external reuse, no execution)."""

from src.grpo.rewards import build_reward
from src.grpo.rewards.library_usage import LibraryUsageReward


# every external-reuse component on, so each test can read off which fired
_WEIGHTS = {
    "external_lib_imported": 1.0,
    "external_lib_used": 1.0,
    "appropriate_lib_used": 1.0,
}


def _completion(code: str) -> str:
    """Wrap python code as a completion with reasoning and a fenced block."""
    return f"reasoning</think>\n```python\n{code}\n```"


class TestLibraryUsageReward:
    """Tests for LibraryUsageReward.score."""

    def test_used_external_library_earns_import_use_and_bonus(self):
        """A used, appropriate third-party import fires all three components."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nrequests.get('http://x')"),
            {"local_modules": [], "useful_libs": ["requests"]},
        )

        assert score == 3.0

    def test_external_library_not_in_useful_list_skips_the_bonus(self):
        """An external lib that isn't an appropriate one earns import+use only."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nrequests.get('http://x')"),
            {"local_modules": [], "useful_libs": ["httpx"]},
        )

        assert score == 2.0

    def test_stdlib_only_earns_nothing(self):
        """Reaching for the standard library is not external reuse here."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import math\nprint(math.sqrt(4))"),
            {"local_modules": [], "useful_libs": []},
        )

        assert score == 0.0

    def test_project_local_import_is_not_external(self):
        """Importing the project's own modules doesn't count as library reuse."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("from app.models import User\nUser()"),
            {"local_modules": ["app"], "useful_libs": []},
        )

        assert score == 0.0

    def test_relative_import_is_not_external(self):
        """Relative imports are the project's own code, never external reuse."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("from . import helpers\nhelpers.run()"),
            {"local_modules": [], "useful_libs": []},
        )

        assert score == 0.0

    def test_imported_but_unused_external_lib_earns_import_only(self):
        """An external import that is never called earns import but not use/bonus."""
        reward = LibraryUsageReward(_WEIGHTS)

        score = reward.score(
            _completion("import requests\nprint('hello')"),
            {"local_modules": [], "useful_libs": ["requests"]},
        )

        assert score == 1.0

    def test_build_reward_selects_this_class(self):
        """The factory resolves the library-usage kind to this reward."""
        reward = build_reward("library_usage", _WEIGHTS)

        assert isinstance(reward, LibraryUsageReward)
