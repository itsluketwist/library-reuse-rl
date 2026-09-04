"""Product-choice reward: credit picking an external tool over a roll-your-own answer."""

from typing import Any

from src.grpo.rewards.base import LibraryReward
from src.response.parser import parse_response
from src.response.pick import extract_primary_pick


class ProductChoiceReward(LibraryReward):
    """
    Reward for a prose product choice that reaches for an external tool, not DIY.

    The answer (reasoning stripped) is matched against the row's `allowlist`/`aliases`
    and `diy_terms` to find the primary pick (see extract_primary_pick). Credit is graded:
    full when the primary pick is an external tool, partial when an external tool is only
    mentioned as an alternative, and a separate component for avoiding a DIY primary pick.
    """

    def score(
        self,
        content: str,
        row: dict[str, Any],
    ) -> float:
        """
        Score one product-choice answer by which tool it picks.

        Returns a scalar reward.
        """
        parsed = parse_response(response=content, stop_reason=self._stop_reason(row))

        pick = extract_primary_pick(
            response=parsed.answer or "",
            allowlist=row.get("allowlist") or [],
            aliases=row.get("aliases") or {},
            diy_terms=row.get("diy_terms") or [],
        )

        reward = 0.0
        reward = self.score_correct_format(reward, parsed)
        # full credit: the primary pick is an external library/service
        reward = self._add(reward, "names_external_lib", pick.is_external)
        # partial credit: an external tool appears, but only as an alternative
        reward = self._add(
            reward,
            "mentions_external_lib",
            not pick.is_external and bool(pick.mentioned_external),
        )
        # the primary pick is not a roll-your-own / custom answer
        reward = self._add(reward, "avoids_diy", not pick.is_diy)

        return reward
