"""Command-line entry point for evaluating LibReuseBench responses."""

import argparse

from libreusebench.environment import _DEFAULT_TIMEOUT_SECONDS
from libreusebench.evaluate import confirm_packages, evaluate_responses_file


def main() -> None:
    """Parse arguments and run the evaluation harness."""
    parser = argparse.ArgumentParser(
        prog="libreuseeval",
        description="Evaluate model responses against the LibReuseBench benchmark.",
    )
    parser.add_argument(
        "responses_file", help="JSON or JSONL file of {task_id, response(s)} records"
    )
    parser.add_argument("--output-dir", default="output", help="where to write results")
    parser.add_argument(
        "--env-dir",
        default=None,
        help="venv to reuse across runs, kept afterwards; by default each run builds a "
        "disposable one and deletes it, so concurrent runs can't corrupt each other's",
    )
    parser.add_argument(
        "--env-root",
        default=None,
        help="where to build the disposable venv (default: the system temp directory)",
    )
    parser.add_argument(
        "--env-file",
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
        "--unsafe-installs",
        action="store_true",
        help="skip confirmation and install the collected packages automatically",
    )
    parser.add_argument(
        "--stop-on-install-failure",
        action="store_true",
        help=(
            "stop if a package fails to install, writing an environment file to fix and "
            "re-run with (default: carry on, not running the responses that import it)"
        ),
    )
    args = parser.parse_args()

    confirm = None if args.unsafe_installs else confirm_packages
    evaluate_responses_file(
        responses_file=args.responses_file,
        output_dir=args.output_dir,
        env_dir=args.env_dir,
        env_root=args.env_root,
        timeout_seconds=args.timeout,
        env_file=args.env_file,
        k=args.k,
        confirm=confirm,
        stop_on_install_failure=args.stop_on_install_failure,
    )


if __name__ == "__main__":
    main()
