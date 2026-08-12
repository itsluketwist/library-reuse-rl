"""Command-line entry point for evaluating BigCodeBenchReuse responses."""

import argparse

from bcbreuse.environment import _DEFAULT_TIMEOUT_SECONDS
from bcbreuse.evaluate import _confirm_packages, evaluate_responses


def main() -> None:
    """Parse arguments and run the evaluation harness."""
    parser = argparse.ArgumentParser(
        prog="bcbeval",
        description="Evaluate model responses against the BigCodeBenchReuse benchmark.",
    )
    parser.add_argument(
        "responses_file", help="JSONL of {task_id, response(s)} records"
    )
    parser.add_argument(
        "--output-directory", default="output", help="where to write results"
    )
    parser.add_argument(
        "--env-dir", default=".bcbeval-venv", help="cached venv for installs"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="path to an environment .txt (one pip package per line); "
        "if omitted, packages are collected from the responses and installed automatically",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="per-run timeout (seconds)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k for pass@k (default: the smallest number of samples any task has)",
    )
    parser.add_argument(
        "--confirm-installs",
        action="store_true",
        help="ask for confirmation before installing the collected packages",
    )
    args = parser.parse_args()

    confirm = _confirm_packages if args.confirm_installs else None
    evaluate_responses(
        responses_file=args.responses_file,
        output_directory=args.output_directory,
        env_dir=args.env_dir,
        timeout_seconds=args.timeout,
        env_file=args.env,
        k=args.k,
        confirm=confirm,
    )


if __name__ == "__main__":
    main()
