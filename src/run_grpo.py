"""The command-line entry point and orchestration for RL training with GRPO."""

import os
import sys
import uuid
from argparse import ArgumentParser
from pathlib import Path

from src.grpo.rewards import load_reward_profile
from src.prompts import PromptStyle
from src.utils.config import (
    DEBUG_MODELS_FILE,
    TRAINED_MODELS_FILE,
    load_full_yaml,
    load_model_settings,
    register_model,
)
from src.utils.log import log, log_header, silence_library_warnings
from src.utils.paths import training_run_dir


# default seed, matching the convention in data/download_skywork.py
DEFAULT_SEED = 42

# tiny model (see config/models.yaml) used to smoke-test --debug runs
DEBUG_MODEL_KEY = "qwen3-debug"


# create the main argument parser
parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model key from config/models.yaml. Not required if --debug is set.",
)

parser.add_argument(
    "--run-id",
    type=str,
    default=None,
    help=(
        "Prefix for the registry keys this run's epoch checkpoints are registered under, "
        "as '{run-id}-epoch{n}' (in config/trained_models.yaml, or "
        "config/debug_models.yaml for a --debug run). Auto-generated if not given; "
        "submit_grpo.sh passes one so the dependent eval jobs can name the models "
        "before training has finished."
    ),
)

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="Path to the dataset directory, containing train/val/test.jsonl files.",
)

parser.add_argument(
    "--output-root",
    type=str,
    default="output",
    help=(
        "Base directory trained models are saved under, one folder per run id (see "
        "trained_model_path). Defaults to local 'output'; submit_grpo.sh points it at "
        "scratch storage, where the large adapters and checkpoints live."
    ),
)

parser.add_argument(
    "-rp",
    "--reward-profile",
    type=str,
    default="full",
    help="Reward profile name within the rewards config file.",
)

parser.add_argument(
    "--rewards-config",
    type=str,
    default="config/rewards.yaml",
    help="Path to rewards config file.",
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for training.",
)

parser.add_argument(
    "--recipe-config",
    type=str,
    default="config/training.yaml",
    help="Path to the GRPO training hyperparameters recipe file.",
)

parser.add_argument(
    "--library-prompt-style",
    type=str,
    choices=["lib", "use", "std", "ext", "base"],
    default="lib",
    help=(
        "System-prompt wording (see src/prompts.py): 'lib' lists the available "
        "libraries, 'use' also pushes the model to prefer them (both need the recipe's "
        "libraries_requirements_file), 'std' pushes the model to prefer the standard "
        "library and 'ext' to prefer any third-party library (both independent of that "
        "file), and 'base' skips the system prompt."
    ),
)

parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help=(
        "Number of training epochs, overriding the recipe's num_train_epochs. Used by "
        "the shorter debug runs (see scripts/submit_debug.sh)."
    ),
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=None,
    help=(
        "Peak learning rate, overriding the recipe's learning_rate - so runs can vary it "
        "without a separate recipe file (see training_jobs in scripts/submit_grpo.sh)."
    ),
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help=(
        "Stop after this many optimizer steps, overriding the recipe's run length. Used "
        "to measure step time and memory on a real model without training it."
    ),
)

parser.add_argument(
    "--resume-save-steps",
    type=int,
    default=None,
    help=(
        "Save a resume checkpoint every this many steps, overriding the recipe's "
        "resume_save_steps. The scale check sets it to 1, so a few steps cover a save "
        "followed by more rollouts (see scripts/submit_debug.sh)."
    ),
)

parser.add_argument(
    "--debug",
    action="store_true",
    help=(
        f"Run a tiny smoke test: defaults the model to '{DEBUG_MODEL_KEY}' if --model "
        "isn't given, and truncates the dataset to a few samples, to check the "
        "training code runs end-to-end."
    ),
)


def trained_model_path(
    run_id: str,
    output_root: str = "output",
) -> Path:
    """
    Compute the directory a training run saves its epoch checkpoints to, named by
    `run_id` (the prefix of the registry keys the models are later looked up by) so the
    folder matches the registry entries.

    output_root defaults to "output", but submit_grpo.sh points it at project storage
    since these files are large; small training metrics go elsewhere (see run_grpo).

    Returns the output directory path.
    """
    return Path(output_root) / run_id


def is_main_process() -> bool:
    """
    Whether this is the first of the processes accelerate launched, one per gpu.

    Logging and registry writes happen once per run, not once per gpu.

    Returns True on rank zero, including when running without accelerate.
    """
    return int(os.environ.get("RANK", "0")) == 0


def generate_run_id(model: str) -> str:
    """
    Generate a run id for a run launched without --run-id.

    Every process accelerate launched must agree on it, or each gpu would save to its own
    folder, so a random id is only drawn on a single process. Across several processes
    the slurm job id is used instead, since every process in the job sees the same one.

    Returns the run id, as "{model}-{slurm job id}" or "{model}-{4 random hex characters}".
    """
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return f"{model}-{uuid.uuid4().hex[:4]}"

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        raise ValueError(
            "Pass --run-id when launching several processes outside slurm, so every "
            "process saves to the same folder."
        )
    return f"{model}-{slurm_job_id}"


def run_grpo(
    model: str | None,
    dataset: str,
    reward_profile: str = "full",
    rewards_config: str = "config/rewards.yaml",
    seed: int = DEFAULT_SEED,
    recipe_config: str = "config/training.yaml",
    library_prompt_style: PromptStyle = "lib",
    debug: bool = False,
    output_root: str = "output",
    run_id: str | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    learning_rate: float | None = None,
    resume_save_steps: int | None = None,
) -> None:
    """
    Run GRPO RL training for the given model and dataset.

    `dataset` is a directory of train/val/test.jsonl files (e.g. data/train/library_usage);
    training runs against train.jsonl and uses val.jsonl for periodic evaluation if the
    recipe enables it. `seed` and `library_prompt_style` override the recipe's fields, as
    do `epochs`/`max_steps`/`learning_rate`/`resume_save_steps` when given.

    If `debug` is set, the dataset is truncated to a few samples (a quick end-to-end
    check) and `model` defaults to DEBUG_MODEL_KEY; otherwise `model` is required.

    Resolves the model and reward profile, runs training (see train_model_with_grpo),
    which saves one checkpoint per epoch under trained_model_path, then registers each
    epoch as "{run_id}-epoch{n}" (run_id generated if not given) so eval can look any
    epoch up by name, recording its base, weights path, and the training knobs
    (`properties`). Real runs register in TRAINED_MODELS_FILE; debug runs in
    DEBUG_MODELS_FILE, and only when given a `run_id`.
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

    # debug runs are only registered under a run id chosen on purpose (e.g. the smoke
    # test's fixed one), so throwaway runs don't pile up in the registry
    register_run = not debug or run_id is not None
    registry_file = DEBUG_MODELS_FILE if debug else TRAINED_MODELS_FILE
    if run_id is None:
        run_id = generate_run_id(model=model)

    # defer the legacy gpu backend so help and configuration imports work on a cpu host.
    from src.grpo.train_model import train_model_with_grpo

    model_settings = load_model_settings(model_key=model)
    reward_weights = load_reward_profile(
        profile=reward_profile, config_file=rewards_config
    )
    if is_main_process():
        log(f"Model path: {model_settings['base']}")
        log(f"Dataset: {dataset}")
        log(f"Reward profile '{reward_profile}': {reward_weights}")
        log(f"Seed: {seed}")
        log(f"Recipe: {recipe_config}")
        log(f"Library prompt style: {library_prompt_style}")
        log(f"Debug: {debug}")

    # the large epoch checkpoints go under output_root (project storage on the HPC, see
    # trained_model_path)
    output_path = trained_model_path(run_id=run_id, output_root=output_root)

    # small training metrics stay in the repo, separate from the large checkpoints, in the
    # run's folder alongside each epoch's eval results (see src/utils/paths.py)
    metrics_path = training_run_dir(base_model=model, run_id=run_id) / "grpo"

    if is_main_process():
        log(f"Checkpoint output path: {output_path}")
        log(f"Metrics path: {metrics_path}")

    checkpoints = train_model_with_grpo(
        model_name_or_path=model_settings["base"],
        dataset_dir=dataset,
        output_dir=output_path,
        metrics_dir=metrics_path,
        reward_profile=reward_profile,
        reward_weights=reward_weights,
        seed=seed,
        recipe_config=recipe_config,
        library_prompt_style=library_prompt_style,
        debug=debug,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        resume_save_steps=resume_save_steps,
    )

    # every process trained; only one of them writes to the shared registry
    if not is_main_process():
        return

    if not register_run:
        log(
            f"Debug run without a --run-id - not registered. Checkpoints: {checkpoints}"
        )
        return

    # a --max-steps run stops before the first epoch ends, so there is nothing to register
    if not checkpoints:
        log("No epoch finished, so no checkpoint to register.")
        return

    # the learning rate the run actually used - the override if given, else the recipe's
    if learning_rate is None:
        learning_rate = load_full_yaml(recipe_config)["learning_rate"]

    # the run id is non-descriptive, so record the knobs to recognise the model later
    for epoch, checkpoint in checkpoints.items():
        properties = {
            # which run and base model this is, so eval files its results in that run's
            # folder (see model_output_dir in src/utils/paths.py)
            "run_id": run_id,
            "base_model": model,
            "dataset": dataset,
            "recipe": recipe_config,
            "reward_profile": reward_profile,
            "seed": seed,
            "library_prompt_style": library_prompt_style,
            "learning_rate": learning_rate,
            "epoch": epoch,
        }
        model_key = f"{run_id}-epoch{epoch}"
        register_model(
            model_key=model_key,
            base=model_settings["base"],
            path=str(checkpoint),
            properties=properties,
            config_file=registry_file,
        )
        log(f"Registered trained model '{model_key}' ({checkpoint}) in {registry_file}")


def main() -> None:
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    silence_library_warnings()

    # --- startup banner (once per run, not once per gpu) ---
    if is_main_process():
        log_header("RL TRAINING")
        log(f"Model: {args.model}")
        log(f"Dataset: {args.dataset}")
        log(f"Recipe: {args.recipe_config}")
        log(f"Reward: {args.rewards_config} [{args.reward_profile}]")
        log(f"Seed: {args.seed}")
        log(f"Library prompt style: {args.library_prompt_style}")
        log(f"Epochs: {args.epochs or '(recipe)'}")
        log(f"Max steps: {args.max_steps or '(recipe)'}")
        log(f"Learning rate: {args.learning_rate or '(recipe)'}")
        log(f"Resume save steps: {args.resume_save_steps or '(recipe)'}")
        log(f"Debug: {args.debug}")
        log(f"Output root: {args.output_root}")
        log(f"Run id: {args.run_id or '(generated)'}")
        log()

    run_grpo(
        model=args.model,
        dataset=args.dataset,
        reward_profile=args.reward_profile,
        rewards_config=args.rewards_config,
        seed=args.seed,
        recipe_config=args.recipe_config,
        library_prompt_style=args.library_prompt_style,
        run_id=args.run_id,
        debug=args.debug,
        output_root=args.output_root,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        resume_save_steps=args.resume_save_steps,
    )

    # --- summary ---
    if is_main_process():
        log_header("COMPLETE")
        log("Done.")


if __name__ == "__main__":
    main()
