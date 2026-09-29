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
from src.grpo.rewards.tested_implementation import TestedImplementationReward


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


class _ComponentCache:
    """
    Judges a batch of completions once, for every component callable to share.

    Each component is its own callable so trl can log it separately (see
    build_reward_functions), but they all need the same underlying judgement - and for the
    tested-implementation reward that judgement runs the generated code. This computes it
    once per batch, and the other callables reuse it.
    """

    def __init__(
        self,
        reward: LibraryReward,
    ) -> None:
        self.reward = reward
        self._batch: tuple[list[str], list[dict[str, Any]]] | None = None
        self._components: list[dict[str, bool]] = []

    def components(
        self,
        contents: list[str],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, bool]]:
        """
        Judge every completion in a batch, reusing the result while the batch repeats.

        Returns one component dict per completion.
        """
        if self._batch != (contents, rows):
            self._components = [
                self.reward.components(content, row)
                for content, row in zip(contents, rows)
            ]
            self._batch = (contents, rows)
        return self._components


class ComponentReward:
    """
    A single reward component, wrapped as its own reward function for GRPOTrainer.

    Returns 1 or 0 per completion, so the mean trl logs for it is simply the rate at
    which completions satisfy that component.
    """

    def __init__(
        self,
        cache: _ComponentCache,
        component: str,
    ) -> None:
        self.cache = cache
        self.component = component
        # trl names each logged reward metric after the callable
        self.__name__ = component

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        **columns: list[Any],
    ) -> list[float]:
        """
        Judge every completion in a GRPO batch on this component alone.

        Returns 1.0 for each completion satisfying it, else 0.0.
        """
        contents = [completion[0]["content"] for completion in completions]
        rows = rows_from_columns(columns, n=len(contents))
        return [
            float(components[self.component])
            for components in self.cache.components(contents, rows)
        ]


def build_reward_functions(
    kind: str,
    weights: dict[str, float],
    eos_token_id: int | None = None,
) -> tuple[list[ComponentReward], list[float]]:
    """
    Build one reward function per component of the profile, for task `kind`.

    trl logs each reward function's mean separately, so splitting the reward this way puts
    every component's rate in the training and validation logs (eg.
    "rewards/external_lib_used/mean" is the fraction of completions that reused a
    library). trl multiplies each function by its weight and sums them, so the total
    reward is identical to scoring with build_reward.

    A zero-weighted component is included too: it contributes nothing to the total but is
    still worth watching (the lu-v1 profile sets appropriate_lib_used to 0).

    Returns the reward functions and their matching weights, in the same order.
    """
    reward = build_reward(kind=kind, weights=weights, eos_token_id=eos_token_id)
    cache = _ComponentCache(reward)
    components = sorted(weights)
    return (
        [ComponentReward(cache, component) for component in components],
        [weights[component] for component in components],
    )


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
    "ComponentReward",
    "LibraryReward",
    "LibraryUsageReward",
    "ProductChoiceReward",
    "TestedImplementationReward",
    "build_reward",
    "build_reward_functions",
    "infer_reward_kind",
    "load_reward_profile",
    "rows_from_columns",
]
