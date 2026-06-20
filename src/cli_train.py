"""The command line interface entry-point for RL training."""

from argparse import ArgumentParser

from src.response.reward import ALL_REWARD_BITS, RewardBit, parse_reward_bits
from src.run_train import DEFAULT_SEED, run
from src.utils.log import log, log_header


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
    "--decoding-profile",
    type=str,
    default="greedy",
    help="Decoding profile name within the decoding config file.",
)

parser.add_argument(
    "--decoding-config",
    type=str,
    default="config/decoding.yaml",
    help="Path to decoding config file.",
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


def main() -> None:
    args = parser.parse_args()

    # --- startup banner ---
    log_header("RL TRAINING")
    log(f"Model: {args.model}")
    log(f"Dataset: {args.dataset}")
    log(f"Decoding: {args.decoding_config} [{args.decoding_profile}]")
    log(f"Reward bits: {RewardBit(args.reward_bits)!r}")
    log(f"Seed: {args.seed}")
    log()

    run(
        model=args.model,
        dataset=args.dataset,
        model_config=args.model_config,
        decoding_profile=args.decoding_profile,
        decoding_config=args.decoding_config,
        reward_bits=args.reward_bits,
        seed=args.seed,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
