"""Set the wandb environment variables GRPOTrainer's wandb integration reads."""

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING


# only for the type hint: the config imports trl, which the core environment doesn't have,
# and this module is tested there
if TYPE_CHECKING:
    from src.grpo.config import GRPOConfig


# the file in a run's output directory (beside its checkpoints) holding its wandb run id
WANDB_RUN_ID_FILE = "wandb_run_id.txt"


def init_wandb_training(training_args: "GRPOConfig") -> None:
    """Set the wandb entity/project/group environment variables from `training_args`, and
    the run id, so a requeued job continues the run's own wandb run (see resume_wandb_run).
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

    # only the first gpu's process logs to wandb, so only it reads or writes the id file
    if training_args.process_index == 0:
        resume_wandb_run(training_args.output_dir)


def resume_wandb_run(output_dir: str | Path) -> str:
    """
    Give a training run one wandb run across all of its slurm jobs.

    The first job makes a run id and saves it in the run's output directory; a job requeued
    at the time limit finds it there and continues that wandb run, instead of starting a
    second one with the same name. The id is new for every fresh output directory, so a
    debug run cleaned out and started again gets its own wandb run.

    Returns the wandb run id.
    """
    id_file = Path(output_dir) / WANDB_RUN_ID_FILE
    if id_file.exists():
        run_id = id_file.read_text().strip()
    else:
        run_id = f"{Path(output_dir).name}-{uuid.uuid4().hex[:8]}"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run_id + "\n")

    # read by wandb.init, which transformers' wandb integration calls with no id of its own
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
    return run_id
