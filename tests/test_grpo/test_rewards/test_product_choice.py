"""Tests for the product-choice reward (prose pick, no execution)."""

from src.grpo.rewards import build_reward
from src.grpo.rewards.product_choice import ProductChoiceReward


# the databases-style reference data most tests reuse
_ROW = {
    "allowlist": ["postgresql", "mongodb"],
    "aliases": {"postgresql": ["postgres"]},
    "diy_terms": ["json file"],
}

# every product-choice component on, to read off which fired
_WEIGHTS = {
    "names_external_lib": 1.0,
    "mentions_external_lib": 0.5,
    "avoids_diy": 0.2,
}


class TestProductChoiceReward:
    """Tests for ProductChoiceReward.score."""

    def test_primary_external_pick_earns_full_credit(self):
        """A leading external pick fires names_external_lib and avoids_diy, not mentions."""
        reward = ProductChoiceReward(_WEIGHTS)

        score = reward.score(
            "reasoning</think>\nUse PostgreSQL here; MongoDB is an alternative.",
            _ROW,
        )

        # names_external_lib (1.0) + avoids_diy (0.2); mentions is only for alternatives
        assert score == 1.2

    def test_external_only_as_alternative_earns_partial_credit(self):
        """When DIY leads and a tool trails, mentions fires but names does not."""
        reward = ProductChoiceReward(_WEIGHTS)

        score = reward.score(
            "reasoning</think>\nJust write to a JSON file. If you must, Postgres works.",
            _ROW,
        )

        # primary is DIY: names 0, avoids_diy 0, but Postgres is mentioned -> 0.5
        assert score == 0.5

    def test_negated_tool_is_not_a_pick(self):
        """A tool named only to reject it ('don't use X') is not credited."""
        reward = ProductChoiceReward({"names_external_lib": 1.0})

        score = reward.score(
            "reasoning</think>\nDon't use MongoDB. There's no clear winner.",
            _ROW,
        )

        assert score == 0.0

    def test_correct_format_needs_reasoning_and_answer(self):
        """correct_format fires only when reasoning precedes a non-empty answer."""
        reward = ProductChoiceReward({"correct_format": 1.0})

        assert reward.score("reasoning</think>\nUse PostgreSQL.", _ROW) == 1.0
        # no reasoning tag -> empty reasoning -> not correctly formatted
        assert reward.score("Use PostgreSQL.", _ROW) == 0.0

    def test_build_reward_selects_this_class(self):
        """The factory resolves the product-choice kind to this reward."""
        reward = build_reward("product_choice", _WEIGHTS)

        assert isinstance(reward, ProductChoiceReward)
