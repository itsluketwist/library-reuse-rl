"""Shared GRPO training utilities, with the HPC-only config imported lazily."""

from typing import Any

from .rewards import (
    REWARD_COMPONENTS,
    build_reward,
    compute_reward,
    infer_reward_kind,
    load_reward_profile,
)


__all__ = [
    "GRPOConfig",
    "REWARD_COMPONENTS",
    "build_reward",
    "compute_reward",
    "infer_reward_kind",
    "load_reward_profile",
]


# trl is only installed with the hpc extra; pure reward/scoring modules must remain
# importable in the core development environment used by the test suite.
def __getattr__(name: str) -> Any:
    if name == "GRPOConfig":
        from .config import GRPOConfig

        return GRPOConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
