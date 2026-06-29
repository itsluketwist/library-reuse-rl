"""The command-line entry point and orchestration for evaluating untrained, pretrained, or trained models."""

from argparse import ArgumentParser
from pathlib import Path

from src.eval import evaluate_model
from src.utils.config import load_full_yaml, load_yaml_config
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


def eval_results_path(
    model: str,
    dataset: str,
    decoding_profile: str,
) -> Path:
    """
    Compute the output path an evaluation run saves its results to.

    The decoding profile is part of the path (not just the model and dataset) since
    different decoding settings produce genuinely different results - this keeps
    every run's output uniquely and unambiguously named.

    Returns the output file path.
    """
    return Path("output") / "eval" / f"{model}-{dataset}-{decoding_profile}.jsonl"


def run_eval(
    model: str,
    datasets: list[str] | None = None,
    model_config: str = "config/models.yaml",
    decoding_profile: str = "greedy",
    decoding_config: str = "config/decoding.yaml",
    eval_datasets_config: str = "config/eval_datasets.yaml",
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key from `model_config` (see config/models.yaml), which holds both
    base models (with a blank "path") and trained models (registered by register_model
    in src/utils/config.py once training produces one) - this is what lets the
    post-training eval job (queued by scripts/submit_grpo.sh) point at the model a
    training run just produced, by name, the same way it points at a base model.

    `datasets` is a list of nice names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml). If not given, every dataset in that file is run.

    Resolves the model, decoding settings, and dataset files/output paths, then runs
    evaluation (see src/eval/evaluate.py::evaluate_model, which loads the model and
    handles generation, scoring, and saving itself) against each dataset, saving
    results under the paths returned by `eval_results_path`.
    """
    model_settings = load_yaml_config(config_file=model_config, key=model)
    # prefer a trained checkpoint's path if one's registered, otherwise the base model
    model_name_or_path = model_settings["path"] or model_settings["base"]
    log(f"Model: {model_name_or_path}")

    decoding_settings = load_yaml_config(
        config_file=decoding_config, key=decoding_profile
    )
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")

    available_datasets = load_full_yaml(eval_datasets_config)
    dataset_names = datasets if datasets is not None else list(available_datasets)
    dataset_files = {name: available_datasets[name] for name in dataset_names}
    output_paths = {
        name: eval_results_path(
            model=model, dataset=name, decoding_profile=decoding_profile
        )
        for name in dataset_names
    }

    evaluate_model(
        model_name_or_path=model_name_or_path,
        dataset_files=dataset_files,
        decoding_settings=decoding_settings,
        output_paths=output_paths,
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

    run_eval(
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
