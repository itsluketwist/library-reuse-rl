"""The RL training loop - a placeholder, since the RL algorithm isn't decided yet."""

from pathlib import Path

from src.response.reward import ALL_REWARD_BITS, RewardBit
from src.utils.config import load_yaml_config
from src.utils.log import log


# used whenever a caller doesn't pass an explicit seed - keeps runs reproducible by
# default, matching the convention used in data/download_skywork.py
DEFAULT_SEED = 42


def trained_model_path(
    model: str,
    dataset: str,
    decoding_profile: str,
    reward_bits: RewardBit,
    seed: int,
) -> Path:
    """
    Compute the output path a training run saves its trained model to.

    The decoding profile, reward bits, and seed are all part of the path (not just the
    model and dataset) since each produces a genuinely different trained model - this
    keeps every run's output uniquely and unambiguously named.

    This is deterministic, so a slurm submission script can compute the same path and
    queue the post-training eval job (pointed at this path, rather than the original
    base model) before training has even started running - see scripts/submit_train.sh.

    Returns the output directory path.
    """
    dataset_name = Path(dataset).name
    return Path("output") / (
        f"{model}-{dataset_name}-{decoding_profile}-rb{int(reward_bits)}-seed{seed}"
    )


def run(
    model: str,
    dataset: str,
    model_config: str = "config/models.yaml",
    decoding_profile: str = "greedy",
    decoding_config: str = "config/decoding.yaml",
    reward_bits: RewardBit = ALL_REWARD_BITS,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Run RL training for the given model and dataset.

    `dataset` is a directory containing train/val/test.jsonl files (e.g. data/skywork)
    - training runs against its train.jsonl.

    Placeholder: resolves the model, decoding config, and dataset file, then just
    reports them. Will eventually load the dataset, seed every source of randomness
    with `seed`, run the chosen RL algorithm against the model, and use `reward_bits`
    (see RewardBit in src/response/reward.py) to compute rewards for each response,
    saving the trained output under the path returned by `trained_model_path`.

    Once a trained output actually exists, it should be registered (see register_model
    in src/utils/config.py) in `model_config` under the same name as its output
    directory, so eval can look it up by name just like a base model - not done yet,
    since there's no real output to register.
    """
    model_settings = load_yaml_config(config_file=model_config, key=model)
    log(f"Model path: {model_settings['base']}")

    decoding_settings = load_yaml_config(
        config_file=decoding_config, key=decoding_profile
    )
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")

    train_file = Path(dataset) / "train.jsonl"
    log(f"Dataset: {train_file}")
    log(f"Reward bits: {RewardBit(reward_bits)!r}")
    log(f"Seed: {seed}")

    output_path = trained_model_path(
        model=model,
        dataset=dataset,
        decoding_profile=decoding_profile,
        reward_bits=reward_bits,
        seed=seed,
    )
    log(f"Output model path: {output_path}")
