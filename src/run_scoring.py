"""The command-line entry point for scoring externally generated responses (e.g. Fireworks)."""

from argparse import ArgumentParser
from pathlib import Path

from src.eval.score_generations import score_generations
from src.utils.config import load_full_yaml
from src.utils.log import log, log_header


parser = ArgumentParser(description=__doc__)

parser.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="Dataset name from --eval-datasets-config; its path, evaluator, and execution "
    "environment settings are reused from there, same as run_eval.",
)

parser.add_argument(
    "-g",
    "--generations",
    required=True,
    type=Path,
    help="JSON or JSONL file of externally generated responses, one record per "
    "(task_id, sample_index) with at least task_id, sample_index, content, and "
    "finish_reason (see fireworks/fw_rlvr/inference.py's generation records).",
)

parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Where to save the scored results. Defaults alongside --generations, "
    "replacing its suffix with '.json'.",
)

parser.add_argument(
    "--eval-datasets-config",
    type=str,
    default="config/eval_datasets.yaml",
    help="Path to eval datasets config file.",
)


def main() -> None:
    args = parser.parse_args()

    available_datasets = load_full_yaml(args.eval_datasets_config)
    if args.dataset not in available_datasets:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'; available: {sorted(available_datasets)}"
        )
    dataset_config = available_datasets[args.dataset]

    output = args.output or args.generations.with_suffix(".json")
    environment_root = dataset_config.get("environment_root")
    environment_root = (
        str(Path(environment_root) / output.parent.name / output.stem)
        if environment_root is not None
        else None
    )

    log_header("SCORE GENERATIONS")
    log(f"Dataset: {args.dataset} ({dataset_config['path']})")
    log(f"Evaluator: {dataset_config['evaluator']}")
    log(f"Generations: {args.generations}")
    log(f"Output: {output}")
    log()

    score_generations(
        dataset_file=dataset_config["path"],
        generations_file=args.generations,
        output_file=output,
        evaluator=dataset_config["evaluator"],
        python_executable=dataset_config.get("python_executable"),
        environment_root=environment_root,
    )

    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
