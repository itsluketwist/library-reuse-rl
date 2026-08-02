"""The command-line entry point and orchestration for RL training with GRPO."""

import uuid
from argparse import ArgumentParser
from pathlib import Path

import unsloth  # noqa: F401, E402

from src.grpo.rewards import load_reward_profile
from src.grpo.train_model import train_model_with_grpo
from src.prompts import PromptStyle
from src.utils.config import load_yaml_config, register_model
from src.utils.log import log, log_header


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
    help="Model key from models.yaml. Not required if --debug is set.",
)

parser.add_argument(
    "--run-id",
    type=str,
    default=None,
    help=(
        "Registry key this run's trained model is registered under (see "
        "config/models.yaml). Auto-generated if not given; submit_grpo.sh passes one "
        "so the dependent eval job can find the model without recomputing it."
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
    "-mc",
    "--model-config",
    type=str,
    default="config/models.yaml",
    help="Path to model config file.",
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
    choices=["lib", "use", "std"],
    default="lib",
    help=(
        "System-prompt wording (see src/prompts.py): 'lib' lists the available "
        "libraries, 'use' also pushes the model to prefer them (both need the recipe's "
        "libraries_requirements_file), 'std' pushes the model to prefer the standard "
        "library (independent of that file). Sets the output suffix."
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
    Compute the directory a training run saves its adapter and checkpoints to, named by
    `run_id` (the registry key the model is later looked up by) so the folder matches
    the registry entry.

    output_root defaults to "output", but submit_grpo.sh points it at scratch storage
    since these files are large; small training metrics go elsewhere (see run_grpo).

    Returns the output directory path.
    """
    return Path(output_root) / run_id


def run_grpo(
    model: str | None,
    dataset: str,
    model_config: str = "config/models.yaml",
    reward_profile: str = "full",
    rewards_config: str = "config/rewards.yaml",
    seed: int = DEFAULT_SEED,
    recipe_config: str = "config/training.yaml",
    library_prompt_style: PromptStyle = "lib",
    debug: bool = False,
    output_root: str = "output",
    run_id: str | None = None,
) -> None:
    """
    Run GRPO RL training for the given model and dataset.

    `dataset` is a directory of train/val/test.jsonl files (e.g. data/skywork); training
    runs against train.jsonl and uses val.jsonl for periodic evaluation if the recipe
    enables it. `seed` and `library_prompt_style` override the recipe's fields.

    If `debug` is set, the dataset is truncated to a few samples (a quick end-to-end
    check) and `model` defaults to DEBUG_MODEL_KEY; otherwise `model` is required.

    Resolves the model and reward profile, runs training (see train_model_with_grpo),
    saves the adapter under trained_model_path, then registers the trained model in
    `model_config` under `run_id` (generated if not given) so eval can look it up by
    name, recording its base, adapter path, and the training knobs (`properties`).
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

    if run_id is None:
        run_id = f"{model}-{uuid.uuid4().hex[:4]}"

    model_settings = load_yaml_config(config_file=model_config, key=model)
    reward_weights = load_reward_profile(
        profile=reward_profile, config_file=rewards_config
    )
    log(f"Model path: {model_settings['base']}")
    log(f"Dataset: {dataset}")
    log(f"Reward profile '{reward_profile}': {reward_weights}")
    log(f"Seed: {seed}")
    log(f"Recipe: {recipe_config}")
    log(f"Library prompt style: {library_prompt_style}")
    log(f"Debug: {debug}")

    # large adapter/checkpoints go under output_root (scratch storage, see trained_model_path)
    output_path = trained_model_path(run_id=run_id, output_root=output_root)
    log(f"Adapter output path: {output_path}")

    # small training metrics stay in the repo, separate from the large scratch artifacts
    metrics_path = Path("output") / "grpo" / run_id
    log(f"Metrics path: {metrics_path}")

    train_model_with_grpo(
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
    )

    # debug runs are throwaway, so don't register them
    if debug:
        eval_command = (
            f"run_eval --debug --model {model} --adapter {output_path} "
            f"--decoding-config {recipe_config} --prompt-style {library_prompt_style}"
        )
        log(
            f"Debug run - not registered in {model_config}. Evaluate this adapter with:"
        )
        log(f"  {eval_command}")
        return

    # the run id is non-descriptive, so record the knobs to recognise the model later
    properties = {
        "dataset": dataset,
        "recipe": recipe_config,
        "reward_profile": reward_profile,
        "seed": seed,
        "library_prompt_style": library_prompt_style,
    }
    register_model(
        model_key=run_id,
        base=model_settings["base"],
        path=str(output_path),
        properties=properties,
        config_file=model_config,
    )
    log(f"Registered trained model '{run_id}' ({output_path}) in {model_config}")


def main() -> None:
    args = parser.parse_args()

    # --- startup banner ---
    log_header("RL TRAINING")
    log(f"Model: {args.model}")
    log(f"Dataset: {args.dataset}")
    log(f"Recipe: {args.recipe_config}")
    log(f"Reward: {args.rewards_config} [{args.reward_profile}]")
    log(f"Seed: {args.seed}")
    log(f"Library prompt style: {args.library_prompt_style}")
    log(f"Debug: {args.debug}")
    log(f"Output root: {args.output_root}")
    log(f"Run id: {args.run_id or '(generated)'}")
    log()

    run_grpo(
        model=args.model,
        dataset=args.dataset,
        model_config=args.model_config,
        reward_profile=args.reward_profile,
        rewards_config=args.rewards_config,
        seed=args.seed,
        recipe_config=args.recipe_config,
        library_prompt_style=args.library_prompt_style,
        run_id=args.run_id,
        debug=args.debug,
        output_root=args.output_root,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
