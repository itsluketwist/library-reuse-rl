"""CLI adapter from Fireworks JSONL generations to existing project evaluation."""

import argparse
from pathlib import Path

from fw_rlvr.scoring import score_generation_file


def main() -> None:
    """Parse scoring arguments and run the repository's configured scorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluator", default="library_usage")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--environment-dir", default=None)
    args = parser.parse_args()
    output = score_generation_file(
        dataset_file=args.dataset,
        generations_file=args.generations,
        output_file=args.output,
        evaluator=args.evaluator,
        python_executable=args.python_executable,
        environment_dir=args.environment_dir,
    )
    print(f"Scored {len(output['results'])} response(s) into {args.output}.")


if __name__ == "__main__":
    main()
