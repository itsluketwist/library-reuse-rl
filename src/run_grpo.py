"""The command-line entry point and orchestration for RL training with GRPO."""

from argparse import ArgumentParser
from pathlib import Path

from src.grpo.rewards import load_reward_profile
from src.grpo.train_model import train_model_with_grpo
from src.prompts import PromptStyle, library_prompt_suffix
from src.utils.config import load_full_yaml, load_yaml_config, register_model
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
    default="config/train/demo.yaml",
    help="Path to the GRPO training hyperparameters recipe file.",
)

parser.add_argument(
    "--library-prompt-style",
    type=str,
    choices=["lib", "use"],
    default="lib",
    help=(
        "Wording used for the library system prompt when the recipe's "
        "libraries_requirements_file is set (see "
        "src/prompts.py::build_system_prompt): 'lib' just lists the available "
        "libraries, 'use' also pushes the model to prefer them over writing its own "
        "implementations. Ignored when the recipe has no libraries_requirements_file. "
        "Changes the output path suffix from e.g. '-libv1' to '-usev1'."
    ),
)

parser.add_argument(
    "--debug",
    action="store_true",
    help=(
        f"Run a tiny smoke test: defaults the model to '{DEBUG_MODEL_KEY}' (see "
        "config/models.yaml) if --model isn't given, and truncates the dataset to a "
        "handful of samples, to quickly check the training code runs end-to-end."
    ),
)


def trained_model_path(
    model: str,
    dataset: str,
    recipe_config: str,
    reward_profile: str,
    seed: int,
    library_prompt_style: PromptStyle = "lib",
) -> Path:
    """
    Compute the output path a training run saves its trained model to.

    Output lives under output/{model}/ so all runs for the same base model stay
    together. The recipe config, reward profile, and seed (when non-default) are
    encoded in the run name. When the recipe sets libraries_requirements_file, a
    suffix (e.g. "-libv1" or "-usev1", depending on `library_prompt_style`) is also
    appended, so runs under different library system-prompt settings stay separate.
    `library_prompt_style` is a CLI-level setting (see --library-prompt-style below),
    not part of the recipe file, mirroring how `seed` overrides the recipe.

    This naming convention is deterministic, so scripts/submit_grpo.sh mirrors it in
    bash (python/the venv aren't available on the submission node) to compute the same
    path and queue the post-training eval job before training has even started - keep
    that script in sync if this naming convention changes.

    Returns the output directory path.
    """
    recipe = load_full_yaml(recipe_config)
    libraries_requirements_file = recipe.get("libraries_requirements_file")

    dataset_name = Path(dataset).name
    recipe_name = Path(recipe_config).stem
    # only include the seed when it differs from the default, to keep names short
    seed_suffix = f"-seed{seed}" if seed != DEFAULT_SEED else ""
    lib_suffix = library_prompt_suffix(
        libraries_requirements_file, style=library_prompt_style
    )
    run_name = f"{dataset_name}-{recipe_name}-{reward_profile}{seed_suffix}{lib_suffix}"
    return Path("output") / model / run_name


def run_grpo(
    model: str | None,
    dataset: str,
    model_config: str = "config/models.yaml",
    reward_profile: str = "full",
    rewards_config: str = "config/rewards.yaml",
    seed: int = DEFAULT_SEED,
    recipe_config: str = "config/train/demo.yaml",
    library_prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Run GRPO RL training for the given model and dataset.

    `dataset` is a directory containing train/val/test.jsonl files (e.g. data/skywork)
    - training runs against its train.jsonl, and uses val.jsonl for periodic
    evaluation if the recipe at `recipe_config` switches that on.

    `library_prompt_style` overrides the recipe's library_prompt_style the same way
    `seed` overrides its seed field (see src/grpo/config.py::GRPOConfig) - only takes
    effect when the recipe sets libraries_requirements_file.

    If `debug` is set, src/grpo/train_model.py::train_model_with_grpo truncates the
    dataset to a handful of samples - useful to quickly check the training code runs
    end-to-end. `model` defaults to `DEBUG_MODEL_KEY` in that case, but an explicit
    `model` is still respected, so a bigger model can be smoke-tested too. Without
    `debug`, `model` is required.

    Resolves the model and reward profile, then runs GRPO training (see
    src/grpo/train_model.py::train_model_with_grpo, which loads the dataset itself -
    see src/data.py::load_train_dataset) with the weights from `reward_profile` (see
    config/rewards.yaml and src/grpo/rewards.py) selecting which reward components
    count, and saves the trained output under the path returned by
    `trained_model_path`. Once training finishes, registers the trained model in
    `model_config` under that output path's name, so eval can look it up by name just
    like a base model.
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

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

    output_path = trained_model_path(
        model=model,
        dataset=dataset,
        recipe_config=recipe_config,
        reward_profile=reward_profile,
        seed=seed,
        library_prompt_style=library_prompt_style,
    )
    log(f"Output model path: {output_path}")

    train_model_with_grpo(
        model_name_or_path=model_settings["base"],
        dataset_dir=dataset,
        output_dir=output_path,
        reward_profile=reward_profile,
        reward_weights=reward_weights,
        seed=seed,
        recipe_config=recipe_config,
        library_prompt_style=library_prompt_style,
        debug=debug,
    )

    # use the path relative to output/ (e.g. "qwen3-8b/skywork-demo-full") as the
    # registry key, so it stays unique across different base models
    model_key = str(output_path.relative_to(Path("output")))
    register_model(
        model_key=model_key,
        base=model_settings["base"],
        path=str(output_path),
        config_file=model_config,
    )
    log(f"Registered trained model '{model_key}' in {model_config}")


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
        debug=args.debug,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
