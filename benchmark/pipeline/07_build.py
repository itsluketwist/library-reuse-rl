"""Stage 07: assemble the author-approved problems into the final eval splits.

Reshapes each surviving problem into the shipped row schema (task, tests, a nested ground_truth,
and a source provenance block) and writes it straight into the libreusebench package's data
directory, where it is bundled and committed as the benchmark.

--track reuse (default) ships the dual-solvable pool: a small dev split is held out and the rest
is the test set, with a dataset-unique task_id assigned after shuffling (LRB/T000... test,
LRB/D000... dev), and ground_truth carries both gold solutions.

--track control ships the no-reuse-expected pool as a single split (LRB/C000...), since there is no
library-reuse signal to hold a dev set out for; ground_truth carries only the stdlib gold, as these
problems have no library solution.

Only problems both stage-06 reviewers kept (or updated) ship - reject wins on any disagreement, see
`review.accepted_ids` - and the build refuses to run while any validated problem is still missing a
verdict from either reviewer, so the released set is always fully double human-checked.

Usage: python benchmark/pipeline/07_build.py [--track reuse|control]
"""

import argparse
import random

from build import DATA_DIR
from review import approved_problems, pending, validated_problems

from src.utils.io import save_jsonl


SEED = 42
DEV_COUNT = 10


def _shape(
    problem: dict,
    track: str,
) -> dict:
    """Reshape a validated problem into the shipped row (without its task_id, added later).

    Returns the row dict.
    """
    ground_truth = {"gold_stdlib": problem["gold_stdlib"]}
    if track == "reuse":
        # the external-library solution plus the version-pinned packages to run it
        ground_truth |= {
            "gold_extlib": problem["gold_extlib"],
            "gold_extlib_packages": problem["gold_extlib_packages"],
        }
    return {
        "category": problem["category"],
        "prompt": problem["prompt"],
        "test_inputs": problem["inputs"],
        "expected_outputs": problem["expected_outputs"],
        "ground_truth": ground_truth,
        # provenance: the originating BigCodeBench task and the libraries its solution used
        "source": {
            "seed_id": problem["seed_id"],
            "seed_libs": problem["seed_libs"],
        },
    }


def main() -> None:
    """Build a LibReuseBench split from one track's author-approved problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["reuse", "control"],
        default="reuse",
        help="reuse: ship test/dev splits (LRB/T.../LRB/D...); "
        "control: ship a single split (LRB/C...)",
    )
    args = parser.parse_args()
    track = args.track

    problems = validated_problems(track)

    # every validated problem must have a verdict from both reviewers - the shipped set is fully
    # double human-reviewed
    unreviewed = pending(problems, track)
    if unreviewed:
        examples = [problem["seed_id"] for problem in unreviewed[:5]]
        raise SystemExit(
            f"{len(unreviewed)} of {len(problems)} problems still need review "
            f"(run stage 06 first), e.g. {examples}"
        )

    # approved_problems applies any author-edited prompt (an "update" verdict) before we ship
    approved = approved_problems(track)
    rows = [_shape(problem, track) for problem in approved]

    # shuffle once with a fixed seed, for reproducible task_id assignment
    rng = random.Random(SEED)
    rng.shuffle(rows)

    if track == "control":
        control = [{"task_id": f"LRB/C{i:03d}"} | row for i, row in enumerate(rows)]
        save_jsonl(control, DATA_DIR / "control.jsonl")
        print(
            f"built {len(rows)} approved problems ({len(problems) - len(approved)} rejected) "
            f"-> control {len(control)} in {DATA_DIR}/"
        )
        return

    # reuse track: peel off a small dev split, the rest is the test set
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
