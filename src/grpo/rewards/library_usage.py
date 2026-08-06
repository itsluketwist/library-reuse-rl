"""Library-usage reward: credit reusing an external library in code, no execution."""

from typing import Any

from src.grpo.rewards.base import CodeReward


class LibraryUsageReward(CodeReward):
    """
    Reward for reaching for an external library instead of reimplementing, in code.

    Verified programmatically (no execution): the completion's imports are classified,
    the standard library and the project's own modules (`local_modules`) are subtracted,
    and anything left that is actually used counts as genuine library reuse. An optional
    bonus fires when the used library is one of the task's `useful_libs` - an appropriate
    choice for the category - but the base reuse signal never depends on that list.
    """

    def score(
        self,
        content: str,
        row: dict[str, Any],
    ) -> float:
        """
        Score one completion by its use of an external library.

        Returns a scalar reward.
        """
        parsed = self.parse(content)

        reward = 0.0
        reward = self.score_format_and_syntax(reward, parsed)

        # external libraries actually reused, excluding the project's own modules
        external = self.external_libraries(parsed, row.get("local_modules"))
        imported = bool(external)
        used = [lib for lib in external if lib in parsed.lib_usage]

        reward = self._add(reward, "external_lib_imported", imported)
        reward = self._add(reward, "external_lib_used", bool(used))

        # bonus: the reused library is an appropriate one for this task's category
        useful = set(row.get("useful_libs") or [])
        reward = self._add(
            reward,
            "appropriate_lib_used",
            any(lib in useful for lib in used),
        )

        return reward
