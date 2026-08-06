"""Shared GRPO training utilities: training config and reward computation."""

from .config import GRPOConfig
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
