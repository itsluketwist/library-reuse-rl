"""Stage 07: assemble the author-approved problems into the final eval splits.

Reshapes each surviving problem into the shipped row schema (task, tests, a nested ground_truth
with both gold solutions, and a source provenance block) and writes them straight into the
libreusebench package's data directory, where they are bundled and committed as the benchmark. A
small dev split is held out; the rest is the test set. Each row gets a dataset-unique task_id
assigned after shuffling: LRB/T000... for the test split, LRB/D000... for the dev split.

Only problems the author kept in stage 06 ship, and the build refuses to run while any validated
problem is still unreviewed - so the released set is always fully human-checked.

Usage: python benchmark/pipeline/07_build.py
"""

import random

from build import DATA_DIR
from review import approved_problems, pending, validated_problems

from src.utils.io import save_jsonl


SEED = 42
DEV_COUNT = 10


def _shape(
    problem: dict,
) -> dict:
    """Reshape a validated problem into the shipped row (without its task_id, added later).

    Returns the row dict.
    """
    return {
        "category": problem["category"],
        "prompt": problem["prompt"],
        "test_inputs": problem["inputs"],
        "expected_outputs": problem["expected_outputs"],
        # both reference solutions plus the version-pinned packages to run the external one
        "ground_truth": {
            "gold_stdlib": problem["gold_stdlib"],
            "gold_extlib": problem["gold_extlib"],
            "gold_extlib_packages": problem["gold_extlib_packages"],
        },
        # provenance: the originating BigCodeBench task and the libraries its solution used
        "source": {
            "seed_id": problem["seed_id"],
            "seed_libs": problem["seed_libs"],
        },
    }


def main() -> None:
    """Build the LibReuseBench test/dev splits from the author-approved problems."""
    problems = validated_problems()

    # every validated problem must have a verdict - the shipped set is fully human-reviewed
    unreviewed = pending(problems)
    if unreviewed:
        examples = [problem["seed_id"] for problem in unreviewed[:5]]
        raise SystemExit(
            f"{len(unreviewed)} of {len(problems)} problems still need review "
            f"(run stage 06 first), e.g. {examples}"
        )

    # approved_problems applies any author-edited prompt (an "update" verdict) before we ship
    approved = approved_problems()
    rows = [_shape(problem) for problem in approved]

    # shuffle once with a fixed seed, then peel off a small dev split
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_dev = DEV_COUNT if len(rows) >= DEV_COUNT else len(rows)
    dev_rows, test_rows = rows[:n_dev], rows[n_dev:]

    # assign the dataset-unique task_id per split, in the (post-shuffle) order they ship
    test = [{"task_id": f"LRB/T{i:03d}"} | row for i, row in enumerate(test_rows)]
    dev = [{"task_id": f"LRB/D{i:03d}"} | row for i, row in enumerate(dev_rows)]

    save_jsonl(test, DATA_DIR / "test.jsonl")
    save_jsonl(dev, DATA_DIR / "dev.jsonl")
    print(
        f"built {len(rows)} approved problems ({len(problems) - len(approved)} rejected) "
        f"-> test {len(test)}, dev {len(dev)} in {DATA_DIR}/"
    )


if __name__ == "__main__":
    main()
