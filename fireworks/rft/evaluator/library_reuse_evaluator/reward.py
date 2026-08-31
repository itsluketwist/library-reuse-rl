"""Calculate a bounded structural library-reuse reward and diagnostics."""

from dataclasses import dataclass
from typing import Any

from .code_extraction import extract_answer
from .library_extraction import analyse_library_usage


@dataclass(frozen=True)
class RewardConfig:
    """Configurable score levels for the initial Fireworks experiment."""

    not_imported: float = 0.0
    imported_not_used: float = 0.25
    imported_and_used: float = 1.0

    def __post_init__(self) -> None:
        scores = (self.not_imported, self.imported_not_used, self.imported_and_used)
        if any(score < 0 or score > 1 for score in scores):
            raise ValueError("Every reward level must be between 0 and 1.")
        if not self.not_imported <= self.imported_not_used <= self.imported_and_used:
            raise ValueError("Reward levels must be monotonically non-decreasing.")


@dataclass(frozen=True)
class RewardResult:
    """Overall reward and evidence exposed as Eval Protocol component metrics."""

    score: float
    reason: str
    parse_success: bool
    external_library_imported: bool
    external_library_used: bool
    appropriate_library_used: bool
    other_external_library: bool
    external_libraries: tuple[str, ...]
    used_external_libraries: tuple[str, ...]
    appropriate_libraries: tuple[str, ...]
    candidate_count: int
    valid_candidate_count: int

    def binary_metrics(self) -> dict[str, float]:
        """Build the bounded component metrics consumed by Eval Protocol.

        Returns metric-name to 0/1 score mappings.
        """
        return {
            "parse_success": float(self.parse_success),
            "external_library_imported": float(self.external_library_imported),
            "external_library_used": float(self.external_library_used),
            "appropriate_library_used": float(self.appropriate_library_used),
            "other_external_library": float(self.other_external_library),
        }


def score_response(
    content: str | None,
    ground_truth: dict[str, Any],
    config: RewardConfig | None = None,
) -> RewardResult:
    """Score final assistant content using structural evidence only.

    Returns a reward in [0, 1] and detailed non-executing diagnostics.
    """
    reward_config = config or RewardConfig()
    extracted = extract_answer(content)
    local_modules = [str(value) for value in ground_truth.get("local_modules") or []]
    useful_libraries = {
        str(value).lower() for value in ground_truth.get("useful_libs") or []
    }

    analyses = [
        analyse_library_usage(candidate.code, local_modules=local_modules)
        for candidate in extracted.candidates
    ]
    valid = [analysis for analysis in analyses if analysis.parse_success]
    external = (
        set().union(
            *(analysis.external_libraries for analysis in valid),
        )
        if valid
        else set()
    )
    used = (
        set().union(
            *(analysis.used_external_libraries for analysis in valid),
        )
        if valid
        else set()
    )
    appropriate = used & useful_libraries
    other = used - useful_libraries

    parse_success = bool(valid)
    if used:
        score = reward_config.imported_and_used
        reason = f"Used external library: {', '.join(sorted(used))}."
    elif external:
        score = reward_config.imported_not_used
        reason = f"Imported but did not use: {', '.join(sorted(external))}."
    elif parse_success:
        score = reward_config.not_imported
        reason = "Parsed Python but found no external-library import."
    else:
        score = reward_config.not_imported
        reason = "No parseable Python final answer."

    return RewardResult(
        score=score,
        reason=reason,
        parse_success=parse_success,
        external_library_imported=bool(external),
        external_library_used=bool(used),
        appropriate_library_used=bool(appropriate),
        other_external_library=bool(other),
        external_libraries=tuple(sorted(external)),
        used_external_libraries=tuple(sorted(used)),
        appropriate_libraries=tuple(sorted(appropriate)),
        candidate_count=len(extracted.candidates),
        valid_candidate_count=len(valid),
    )
