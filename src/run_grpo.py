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
        "Registry key (see config/models.yaml) this run's trained model is "
        "registered under. Generated automatically if not given - "
        "scripts/submit_grpo.sh always passes one explicitly, generated once in "
        "bash and reused for the dependent eval job's --model, so eval doesn't need "
        "to recompute this run's naming convention to find it."
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
        "trained_model_path). Defaults to the local 'output' directory - "
        "scripts/submit_grpo.sh instead points this at scratch storage, where the "
        "adapters and checkpoints live."
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
    run_id: str,
    output_root: str = "output",
) -> Path:
    """
    Compute the directory a training run saves its model adapter and checkpoints to,
    named by `run_id` - the registry key the model is later looked up by (see run_grpo)
    - so the folder on disk matches the registry entry.

    output_root defaults to the local "output" directory, but scripts/submit_grpo.sh
    points it at scratch storage instead, since these adapter/checkpoint files are large.
    Training metrics go elsewhere (output/grpo/{run_id}, see run_grpo), so only the big
    binary artifacts land here. The run id is short and non-descriptive, so the knobs
    behind it are recorded in the registry entry's "properties" instead of the folder
    name (see src/utils/config.py).

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
    count, and saves the trained adapter under the path returned by
    `trained_model_path` (a folder named by `run_id`). Once training finishes,
    registers the trained model in `model_config` under `run_id` (generated if not
    given - see --run-id), recording its base, adapter path, and the training knobs
    (`properties`), so eval can look it up by that id just like a base model is looked
    up by name.
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

    # the adapter and its checkpoints are large, so they go under output_root -
    # scripts/submit_grpo.sh points that at scratch storage (see trained_model_path)
    output_path = trained_model_path(run_id=run_id, output_root=output_root)
    log(f"Adapter output path: {output_path}")

    # training metrics (loss, rewards over time, periodic held-out eval - see
    # src/grpo/train_model.py) are small and worth keeping in the repo, under
    # output/grpo/{run_id}, separate from the large adapter/checkpoints on scratch
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

    # the run id is non-descriptive, so record the knobs this run was launched with -
    # enough to recognise the model later without decoding its id (see register_model)
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
