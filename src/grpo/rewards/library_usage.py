"""Library-usage reward: credit reusing an external library in code, no execution."""

from typing import Any

from lure import components_from_parsed

from src.grpo.rewards.base import CodeReward


class LibraryUsageReward(CodeReward):
    """
    Reward for reaching for an external library instead of reimplementing, in code.

    Verified programmatically by lure (no execution): imports across every answer code
    block are classified, the standard library and the project's own modules
    (`local_modules`) are subtracted, and anything left that is actually used counts as
    genuine library reuse. An optional bonus fires when the used library is one of the
    task's `useful_libs` - an appropriate choice for the category - but the base reuse
    signal never depends on that list.
    """

    def components(
        self,
        content: str,
        row: dict[str, Any],
    ) -> dict[str, bool]:
        """
        Judge one completion by its use of an external library.

        Returns each component name mapped to whether the completion satisfies it.
        """
        parsed = self.parse(content, row)
        # lure's components are exactly this reward's: correct_format,
        # passes_syntax_check, external_lib_imported, external_lib_used and
        # appropriate_lib_used (the bonus, judged against the row's useful_libs)
        return components_from_parsed(
            parsed,
            appropriate_libraries=row.get("useful_libs") or (),
        )
