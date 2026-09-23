"""Library-usage reward: credit reusing an external library in code, no execution."""

from dataclasses import dataclass
from typing import Any

from src.grpo.rewards.base import CodeReward
from src.response.parser import ParsedResponse


@dataclass(frozen=True)
class LibraryUsageAnalysis:
    """External-library evidence aggregated across every answer code block."""

    n_code_blocks: int
    n_valid_code_blocks: int
    external_libs: list[str]
    used_external_libs: list[str]


def analyse_library_usage(
    parsed: ParsedResponse,
    local_modules: list[str] | None = None,
) -> LibraryUsageAnalysis:
    """Analyse every fenced code block for imported and actually used libraries.

    Standard-library, relative, and repository-local imports do not count as external.
    A library is considered used when the code analyser finds usage in the same block
    that imports it; separate examples are not joined into one synthetic program.

    Returns the aggregate library-usage evidence and code-block counts.
    """
    local = set(local_modules or [])
    external: set[str] = set()
    used: set[str] = set()

    for block in parsed.code_blocks:
        block_external = {lib for lib in block.ext_libs if lib not in local}
        external.update(block_external)
        used.update(lib for lib in block_external if lib in block.lib_usage)

    return LibraryUsageAnalysis(
        n_code_blocks=len(parsed.code_blocks),
        n_valid_code_blocks=sum(bool(block.valid) for block in parsed.code_blocks),
        external_libs=sorted(external),
        used_external_libs=sorted(used),
    )


class LibraryUsageReward(CodeReward):
    """
    Reward for reaching for an external library instead of reimplementing, in code.

    Verified programmatically (no execution): imports across every answer code block are
    classified, the standard library and the project's own modules (`local_modules`) are
    subtracted, and anything left that is actually used counts as genuine library reuse.
    An optional bonus fires when the used library is one of the task's `useful_libs` - an
    appropriate choice for the category - but the base reuse signal never depends on that
    list.
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

        # unlike executable tasks, every code example contributes reuse evidence.
        analysis = analyse_library_usage(parsed, row.get("local_modules"))
        used = analysis.used_external_libs
        # bonus: the reused library is an appropriate one for this task's category
        useful = set(row.get("useful_libs") or [])

        return {
            "correct_format": self.has_correct_format(parsed),
            "passes_syntax_check": self.has_valid_syntax(parsed),
            "external_lib_imported": bool(analysis.external_libs),
            "external_lib_used": bool(used),
            "appropriate_lib_used": any(lib in useful for lib in used),
        }
