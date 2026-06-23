"""The command-line entry point and orchestration for RL training with GRPO."""

from argparse import ArgumentParser
from pathlib import Path

from src.grpo.train_model import train_model_with_grpo
from src.response.reward import ALL_REWARD_BITS, RewardBit, parse_reward_bits
from src.utils.config import load_yaml_config, register_model
from src.utils.log import log, log_header


# used whenever a caller doesn't pass an explicit seed - keeps runs reproducible by
# default, matching the convention used in data/download_skywork.py
DEFAULT_SEED = 42


# create the main argument parser
parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Model key from models.yaml.",
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
    "-rb",
    "--reward-bits",
    type=parse_reward_bits,
    default=ALL_REWARD_BITS,
    help=(
        "Comma-separated RewardBit names selecting which reward components are "
        "switched on (see RewardBit in src/response/reward.py). Defaults to every "
        "component switched on. Available bits: "
        + ", ".join(bit.name for bit in RewardBit if bit.name is not None)
    ),
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


def trained_model_path(
    model: str,
    dataset: str,
    recipe_config: str,
    reward_bits: RewardBit,
    seed: int,
) -> Path:
    """
    Compute the output path a training run saves its trained model to.

    The recipe config, reward bits, and seed are all part of the path (not just the
    model and dataset) since each produces a genuinely different trained model - this
    keeps every run's output uniquely and unambiguously named.

    This is deterministic, so a slurm submission script can compute the same path and
    queue the post-training eval job (pointed at this path, rather than the original
    base model) before training has even started running - see scripts/submit_grpo.sh.

    Returns the output directory path.
    """
    dataset_name = Path(dataset).name
    recipe_name = Path(recipe_config).stem
    return Path("output") / (
        f"{model}-{dataset_name}-{recipe_name}-rb{int(reward_bits)}-seed{seed}"
    )


def run(
    model: str,
    dataset: str,
    model_config: str = "config/models.yaml",
    reward_bits: RewardBit = ALL_REWARD_BITS,
    seed: int = DEFAULT_SEED,
    recipe_config: str = "config/train/demo.yaml",
) -> None:
    """
    Run GRPO RL training for the given model and dataset.

    `dataset` is a directory containing train/val/test.jsonl files (e.g. data/skywork)
    - training runs against its train.jsonl, and uses val.jsonl for periodic
    evaluation if the recipe at `recipe_config` switches that on.

    Resolves the model, then runs GRPO training (see
    src/grpo/train_model.py::train_model_with_grpo, which loads the dataset itself -
    see src/data.py::load_train_dataset) with `reward_bits` (see RewardBit in
    src/response/reward.py) selecting which reward components count, and saves the
    trained output under the path returned by `trained_model_path`. Once training
    finishes, registers the trained model in `model_config` under that output path's
    name, so eval can look it up by name just like a base model.
    """
    model_settings = load_yaml_config(config_file=model_config, key=model)
    log(f"Model path: {model_settings['base']}")
    log(f"Dataset: {dataset}")
    log(f"Reward bits: {RewardBit(reward_bits)!r}")
    log(f"Seed: {seed}")
    log(f"Recipe: {recipe_config}")

    output_path = trained_model_path(
        model=model,
        dataset=dataset,
        recipe_config=recipe_config,
        reward_bits=reward_bits,
        seed=seed,
    )
    log(f"Output model path: {output_path}")

    train_model_with_grpo(
        model_name_or_path=model_settings["base"],
        dataset_dir=dataset,
        output_dir=output_path,
        reward_bits=reward_bits,
        seed=seed,
        recipe_config=recipe_config,
    )

    register_model(
        model_key=output_path.name,
        base=model_settings["base"],
        path=str(output_path),
        config_file=model_config,
    )
    log(f"Registered trained model '{output_path.name}' in {model_config}")


def main() -> None:
    args = parser.parse_args()

    # --- startup banner ---
    log_header("RL TRAINING")
    log(f"Model: {args.model}")
    log(f"Dataset: {args.dataset}")
    log(f"Recipe: {args.recipe_config}")
    log(f"Reward bits: {RewardBit(args.reward_bits)!r}")
    log(f"Seed: {args.seed}")
    log()

    run(
        model=args.model,
        dataset=args.dataset,
        model_config=args.model_config,
        reward_bits=args.reward_bits,
        seed=args.seed,
        recipe_config=args.recipe_config,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
