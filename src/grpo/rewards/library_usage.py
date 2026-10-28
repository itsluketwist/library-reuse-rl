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
    genuine library reuse. The target components repeat the import and use checks, but
    only count the task's `target_libs` - the libraries listed for its category - so a
    profile can reward reuse of any library, or only of the target ones.
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
        # passes_syntax_check, external_lib_imported, external_lib_used,
        # target_lib_imported and target_lib_used (judged against the row's target_libs)
        return components_from_parsed(
            parsed,
            target_libraries=row.get("target_libs") or (),
        )
