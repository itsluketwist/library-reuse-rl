"""The command line interface entry-point for evaluating untrained or pretrained models."""

from argparse import ArgumentParser

from src.run_eval import run
from src.utils.log import log, log_header


# create the main argument parser
parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help=(
        "Model key from models.yaml - either a base model, or a trained model "
        "registered by register_model in src/utils/config.py."
    ),
)

parser.add_argument(
    "-d",
    "--datasets",
    type=str,
    default=None,
    help=(
        "Comma-separated dataset names from eval_datasets.yaml. "
        "Defaults to every dataset in that file."
    ),
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
    "--eval-datasets-config",
    type=str,
    default="config/eval_datasets.yaml",
    help="Path to eval datasets config file.",
)


def main() -> None:
    args = parser.parse_args()
    datasets = args.datasets.split(",") if args.datasets else None

    # --- startup banner ---
    log_header("EVALUATION")
    log(f"Model: {args.model}")
    log(f"Datasets: {datasets or 'all'}")
    log(f"Decoding: {args.decoding_config} [{args.decoding_profile}]")
    log()

    run(
        model=args.model,
        datasets=datasets,
        model_config=args.model_config,
        decoding_profile=args.decoding_profile,
        decoding_config=args.decoding_config,
        eval_datasets_config=args.eval_datasets_config,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
