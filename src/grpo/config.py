"""This project's extensions to trl's GRPOConfig."""

from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    Extra args this project's training loop reads, on top of trl's own GRPOConfig.
    """

    libraries_requirements_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Requirements file (see setup/requirements/) listing the libraries "
                "available in the sandboxed eval environment. When set, the system "
                "prompt lists them (worded per library_prompt_style, see "
                "src/prompts.py) and it's appended to the output path as e.g. "
                "'-libv1'. When null, no system prompt is used."
            ),
        },
    )
    library_prompt_style: str = field(
        default="lib",
        metadata={
            "help": (
                "System-prompt wording (see src/prompts.py): 'lib' lists the available "
                "libraries, 'use' also pushes the model to prefer them, 'std' pushes "
                "the model to prefer the standard library and 'ext' to prefer any "
                "third-party library (both independent of libraries_requirements_file), "
                "and 'base' skips the system prompt. "
                "Set via src/run_grpo.py's "
                "--library-prompt-style flag rather than the recipe file."
            ),
        },
    )
    vllm_kv_cache_memory_gib: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Fixed size of the colocated vllm engine's kv cache, in GiB. When set, "
                "vllm skips its startup memory profiling (and so ignores "
                "vllm_gpu_memory_utilization) - see _configure_trl_vllm_engine in "
                "src/grpo/train_model.py for why."
            ),
        },
    )
    resume_save_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Also save a checkpoint every this many optimizer steps, beside each "
                "epoch's, so a run requeued at the time limit loses at most this many "
                "steps of work. Only the newest is kept (see prune_resume_state). Null "
                "saves at epoch ends only."
            ),
        },
    )
    restart_after_save: bool = field(
        default=False,
        metadata={
            "help": (
                "End the training process after every checkpoint save that isn't the "
                "last, for the job script to relaunch it resuming from that checkpoint "
                "(see RestartAfterSaveCallback in src/grpo/train_model.py). A save leaves "
                "gpu memory taken that only a fresh process gets back."
            ),
        },
    )
    split_eval_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Score the dataset's val and test splits with the reward's components "
                "before training and every this many optimizer steps, to track how the "
                "reward generalises (see SplitEvalCallback in src/grpo/train_model.py). "
                "Null turns it off."
            ),
        },
    )
    split_eval_samples: int = field(
        default=5,
        metadata={
            "help": "How many times each val/test prompt is answered in a split eval.",
        },
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": "Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.",
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The entity to store runs under."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The project to store runs under."},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": "The group to store runs under."},
    )
