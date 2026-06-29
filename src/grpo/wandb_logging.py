"""Set the wandb environment variables GRPOTrainer's wandb integration reads."""

import os

from .config import GRPOConfig


def init_wandb_training(training_args: GRPOConfig) -> None:
    """Set the wandb entity/project/group environment variables from `training_args`."""
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
