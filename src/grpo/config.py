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
                "prompt lists those libraries, worded according to "
                "library_prompt_style (see src/prompts.py::build_system_prompt). "
                "When null, no system prompt is used and library_prompt_style is "
                "ignored. Also appended to the output path as e.g. '-libv1'."
            ),
        },
    )
    library_prompt_style: str = field(
        default="lib",
        metadata={
            "help": (
                "Wording used for the library system prompt when "
                "libraries_requirements_file is set: 'lib' just lists the available "
                "libraries, 'use' also pushes the model to prefer them over writing "
                "its own implementations (see src/prompts.py::build_system_prompt). "
                "Changes the output path suffix from e.g. '-libv1' to '-usev1'. "
                "Set via src/run_grpo.py's --library-prompt-style CLI flag rather "
                "than the recipe file, mirroring how --seed overrides seed."
            ),
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
