"""Stage 05: write the benchmark's data files from the tasks that ship.

The splits follow BigCodeBench itself, so that no choice made here shapes them:
- test: every reuse-track task, from a seed whose reference solution uses a third-party library.
- control: every control-track task, from a seed whose reference solution uses only the
  standard library, where no library should help.
Nothing is chosen or capped. Each test row carries a reuse_tier field (low, medium or high, from
how many lines the library saves - see build.reuse_stats) with the line counts behind it, so that
results can be broken down by how much a library helps.

--provisional builds from the tasks the agent review (stage 03) kept, into
benchmark/artefacts/provisional/, so that the benchmark can be evaluated before any human review.
Without it, only tasks both human reviewers approved are used, the splits are written to
benchmark/libreusebench/data/ (where they are bundled as the released benchmark), and the build
refuses to run while any task is still missing a verdict.

--pilot is for evaluating a pilot run. It writes the same splits as --provisional, into
benchmark/artefacts/pilot/, with ids starting LRB-PILOT/, so that a pilot's results are never
mistaken for the benchmark's.

Each task gets a dataset-unique id after a seeded shuffle: LRB/T000... (test) and
LRB/C000... (control).

Usage: python benchmark/pipeline/05_build.py [--provisional | --pilot]
"""

import argparse
import random
import shutil
from collections import Counter

from build import DATA_DIR, PILOT_DIR, PROVISIONAL_DIR, TRACKS, agent_kept_tasks
from review import approved_tasks, pending

from src.utils.io import save_jsonl


SEED = 42

# each split's file name and the letter in its task ids
SPLIT_LETTERS = {"test": "T", "control": "C"}


def by_reuse_value(
    tasks: list[dict],
) -> list[dict]:
    """Order tasks from the highest reuse value to the lowest (ties broken by seed id).

    A control task has no reuse value, so control tasks simply come out in seed order. The order
    only fixes what the seeded shuffle starts from, so that a build is reproducible.

    Returns the ordered tasks.
    """
    return sorted(
        tasks,
        key=lambda task: (-task.get("reuse_value", 0), task["seed_id"]),
    )


def shape(
    task: dict,
) -> dict:
    """Reshape a task into the row the benchmark ships (without its task_id, added later).

    A control task has no library solution, so its row has no library gold and no reuse value.

    Returns the row.
    """
    row = {
        "category": task["category"],
        "domain": task["domain"],
        "prompt": task["prompt"],
        "function_name": task["function_name"],
        "test_inputs": task["test_inputs"],
        "expected_outputs": task["expected_outputs"],
        "ground_truth": {"gold_stdlib": task["stdlib_solution"]},
    }
    # provenance: the BigCodeBench task this one was inspired by, and the libraries that used
    source = {"seed_id": task["seed_id"], "seed_libs": task["seed_libs"]}
    if "extlib_solution" in task:
        # the library solution, with the pinned packages it was checked with
        row["ground_truth"] |= {
            "gold_extlib": task["extlib_solution"],
            "gold_extlib_packages": task["gold_extlib_packages"],
        }
        row |= {
            field: task[field]
            for field in ("reuse_value", "reuse_tier", "stdlib_lloc", "extlib_lloc")
        }
        source["primary_library"] = task["primary_library"]
    return row | {"source": source}


def build_splits(
    reuse_tasks: list[dict],
    control_tasks: list[dict],
    pilot: bool = False,
) -> dict[str, list[dict]]:
    """Put every task that ships in its split, and give each its task id.

    A pilot build uses ids of its own.

    Returns the rows of each split, by split name.
    """
    chosen = {
        "test": by_reuse_value(reuse_tasks),
        "control": by_reuse_value(control_tasks),
    }
    return number_tasks(
        chosen,
        random.Random(SEED),
        prefix="LRB-PILOT" if pilot else "LRB",
    )


def number_tasks(
    chosen: dict[str, list[dict]],
    rng: random.Random,
    prefix: str,
) -> dict[str, list[dict]]:
    """Shuffle each split's tasks, and give each task its id and its shipped shape.

    Returns the rows of each split, by split name.
    """
    splits: dict[str, list[dict]] = {}
    for name, tasks in chosen.items():
        # shuffled, so that a task's id says nothing about its reuse value
        rng.shuffle(tasks)
        splits[name] = [
            {"task_id": f"{prefix}/{SPLIT_LETTERS[name]}{number:03d}"} | shape(task)
            for number, task in enumerate(tasks)
        ]
    return splits


def load_pool(
    track: str,
    provisional: bool,
) -> list[dict]:
    """Load the tasks of a track that may ship.

    A provisional build takes every task the agent review kept. A final build takes only the
    tasks both human reviewers approved, and refuses to go on while any task is unreviewed.

    Returns the task records.
    """
    tasks = agent_kept_tasks(track)
    if provisional:
        return tasks
    unreviewed = pending(tasks, track)
    if unreviewed:
        examples = [task["seed_id"] for task in unreviewed[:5]]
        raise SystemExit(
            f"{len(unreviewed)} of {len(tasks)} {track} tasks still need human review "
            f"(run 04_review.ipynb first), e.g. {examples}"
        )
    return approved_tasks(track, tasks)


def main() -> None:
    """Build every split from the tasks that passed review, and write the data files."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--provisional",
        action="store_true",
        help="build from the agent-kept tasks, before human review, into artefacts/provisional",
    )
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="build from the agent-kept tasks into artefacts/pilot, with pilot ids",
    )
    args = parser.parse_args()

    # only a final build waits for human review
    before_review = args.provisional or args.pilot
    pools = {track: load_pool(track, before_review) for track in TRACKS}
    splits = build_splits(
        reuse_tasks=pools["reuse"],
        control_tasks=pools["control"],
        pilot=args.pilot,
    )

    out_dir = (
        PILOT_DIR if args.pilot else PROVISIONAL_DIR if args.provisional else DATA_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if before_review:
        # the canary marks every folder of benchmark data as eval-only
        shutil.copy(DATA_DIR / "CANARY.txt", out_dir / "CANARY.txt")

    tiers = Counter(task["reuse_tier"] for task in pools["reuse"])
    print(
        f"{len(pools['reuse'])} reuse tasks ("
        + ", ".join(f"{tiers[tier]} {tier}" for tier in ("high", "medium", "low"))
        + f") and {len(pools['control'])} control tasks may ship"
    )
    for name, rows in splits.items():
        if not rows:
            print(f"{name}: no tasks, so no file is written")
            continue
        save_jsonl(rows, out_dir / f"{name}.jsonl")
        print(f"{name}: {len(rows)} tasks -> {out_dir / name}.jsonl")


if __name__ == "__main__":
    main()
