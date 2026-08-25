"""The reward family: one base class overridden per task, plus a selector factory."""

from typing import Any

from src.grpo.rewards.base import (
    REWARD_COMPONENTS,
    CodeReward,
    LibraryReward,
    load_reward_profile,
    rows_from_columns,
)
from src.grpo.rewards.library_usage import LibraryUsageReward
from src.grpo.rewards.product_choice import ProductChoiceReward
from src.grpo.rewards.tested_implementation import (
    TestedImplementationReward,
    compute_reward,
)


# maps a dataset's task type to the reward class that scores it. the tested path is the
# default for the legacy stdin/stdout dataset, which carries no task_type field
_REWARD_CLASSES: dict[str, type[LibraryReward]] = {
    "product_choice": ProductChoiceReward,
    "library_usage": LibraryUsageReward,
    "implement_tested": TestedImplementationReward,
}

# a profile may only contain components its selected reward class actually reads. keeping
# this explicit makes a mismatched dataset/profile fail before an expensive training run.
_SUPPORTED_COMPONENTS: dict[str, frozenset[str]] = {
    "implement_tested": frozenset(
        {
            "correct_format",
            "passes_syntax_check",
            "passed_tests",
            "internal_lib_imported",
            "external_lib_imported",
            "internal_lib_used",
            "external_lib_used",
        }
    ),
    "library_usage": frozenset(
        {
            "correct_format",
            "passes_syntax_check",
            "external_lib_imported",
            "external_lib_used",
            "appropriate_lib_used",
        }
    ),
    "product_choice": frozenset(
        {
            "correct_format",
            "names_external_lib",
            "mentions_external_lib",
            "avoids_diy",
        }
    ),
}

DEFAULT_REWARD_KIND = "implement_tested"


def build_reward(
    kind: str,
    weights: dict[str, float],
    eos_token_id: int | None = None,
) -> LibraryReward:
    """
    Build the reward callable for a task `kind`, bound to `weights`.

    The returned object is callable with GRPOTrainer's (completions, **columns) calling
    convention (see LibraryReward.__call__), so it slots straight into `reward_funcs`.

    Returns the configured reward instance.
    """
    if kind not in _REWARD_CLASSES:
        raise ValueError(
            f"Unknown reward kind '{kind}'. Available: {sorted(_REWARD_CLASSES)}"
        )

    unsupported = sorted(set(weights) - _SUPPORTED_COMPONENTS[kind])
    if unsupported:
        raise ValueError(
            f"Reward profile contains component(s) {unsupported} that '{kind}' does "
            f"not score. Supported components: {sorted(_SUPPORTED_COMPONENTS[kind])}"
        )
    return _REWARD_CLASSES[kind](weights, eos_token_id=eos_token_id)


def infer_reward_kind(
    example: dict[str, Any],
) -> str:
    """
    Infer which reward a dataset row needs from its `task_type` field.

    Rows from the new datasets carry an explicit `task_type`; the legacy stdin/stdout
    dataset has none, so it falls back to the tested-implementation reward.

    Returns the reward kind key (see build_reward).
    """
    return example.get("task_type") or DEFAULT_REWARD_KIND


__all__ = [
    "REWARD_COMPONENTS",
    "CodeReward",
    "LibraryReward",
    "LibraryUsageReward",
    "ProductChoiceReward",
    "TestedImplementationReward",
    "build_reward",
    "compute_reward",
    "infer_reward_kind",
    "load_reward_profile",
    "rows_from_columns",
]
