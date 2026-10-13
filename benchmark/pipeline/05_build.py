"""Stage 05: choose the tasks that ship, split them, and write the benchmark's data files.

The splits are:
- test: the high-tier tasks, where a library saves a lot of code. This is the main split.
- dev: a few high-tier tasks held out of test (DEV_COUNT of them), for trying prompts on.
- low_value: about LOW_VALUE_COUNT low-tier tasks, where a library saves little, spread across
  libraries. They show the contrast with the test split.
- control: the control track's tasks, where no library should help.
Medium-tier tasks are in no split.

Variety is enforced, so that no kind of task and no library dominates:
- at most MAX_PER_CATEGORY tasks of any one category in each pool;
- no library is the primary library of more than MAX_LIBRARY_SHARE of the test split.
Where a cap forces a choice, the tasks with the highest reuse value are kept.

--provisional builds from the tasks the agent review (stage 03) kept, into
benchmark/artefacts/provisional/, so that the benchmark can be evaluated before any human review.
Without it, only tasks both human reviewers approved are used, the splits are written to
benchmark/libreusebench/data/ (where they are bundled as the released benchmark), and the build
refuses to run while any task is still missing a verdict.

--pilot is for evaluating a pilot run, where every task counts. It writes every task the agent
review kept to benchmark/artefacts/pilot/, with no variety caps and no dev split: all the reuse
tasks, of any tier, go to test.jsonl, and all the control tasks to control.jsonl. Their ids start
LRB2-PILOT/, so that a pilot's results are never mistaken for the benchmark's.

Each task gets a dataset-unique id after a seeded shuffle: LRB2/T000... (test), LRB2/D000...
(dev), LRB2/L000... (low_value) and LRB2/C000... (control).

Usage: python benchmark/pipeline/05_build.py [--provisional | --pilot]
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict

from build import DATA_DIR, PILOT_DIR, PROVISIONAL_DIR, TRACKS, agent_kept_tasks
from review import approved_tasks, pending

from src.utils.io import save_jsonl


SEED = 42
DEV_COUNT = 10
LOW_VALUE_COUNT = 40
MAX_PER_CATEGORY = 3
MAX_LIBRARY_SHARE = 0.2

# each split's file name and the letter in its task ids
SPLIT_LETTERS = {"test": "T", "dev": "D", "low_value": "L", "control": "C"}


def by_reuse_value(
    tasks: list[dict],
) -> list[dict]:
    """Order tasks from the highest reuse value to the lowest (ties broken by seed id).

    A control task has no reuse value, so control tasks simply come out in seed order.

    Returns the ordered tasks.
    """
    return sorted(
        tasks,
        key=lambda task: (-task.get("reuse_value", 0), task["seed_id"]),
    )


def cap_categories(
    tasks: list[dict],
) -> list[dict]:
    """Keep at most MAX_PER_CATEGORY tasks of each category, preferring the highest reuse value.

    Returns the tasks kept, from the highest reuse value down.
    """
    kept: list[dict] = []
    counts: Counter[str] = Counter()
    for task in by_reuse_value(tasks):
        if counts[task["category"]] < MAX_PER_CATEGORY:
            counts[task["category"]] += 1
            kept.append(task)
    return kept


def cap_libraries(
    tasks: list[dict],
) -> list[dict]:
    """Drop tasks until no library is the primary library of more than MAX_LIBRARY_SHARE of them.

    The most common library loses its lowest-value task, again and again, until it is within
    the share. Every library may keep one task, so a pool with few libraries is not emptied.

    Returns the tasks kept, from the highest reuse value down.
    """
    kept = by_reuse_value(tasks)
    while kept:
        counts = Counter(task["primary_library"] for task in kept)
        library, count = counts.most_common(1)[0]
        if count <= max(1, int(MAX_LIBRARY_SHARE * len(kept))):
            break
        # the list is ordered by value, so the library's last task is its lowest-value one
        lowest = max(
            index
            for index, task in enumerate(kept)
            if task["primary_library"] == library
        )
        del kept[lowest]
    return kept


def spread_across_libraries(
    tasks: list[dict],
    count: int,
) -> list[dict]:
    """Pick up to `count` tasks, taking one from each library in turn.

    Returns the tasks picked.
    """
    by_library: defaultdict[str, list[dict]] = defaultdict(list)
    for task in by_reuse_value(tasks):
        by_library[task["primary_library"]].append(task)

    picked: list[dict] = []
    while len(picked) < count and any(by_library.values()):
        for library in sorted(by_library):
            if by_library[library] and len(picked) < count:
                picked.append(by_library[library].pop(0))
    return picked


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
    """Choose and split the tasks that ship, and give each its task id.

    A pilot build chooses nothing: every reuse task goes to test, and every control task to
    control, under ids of their own.

    Returns the rows of each split, by split name.
    """
    rng = random.Random(SEED)
    if pilot:
        chosen = {"test": by_reuse_value(reuse_tasks), "control": list(control_tasks)}
        return number_tasks(chosen, rng, prefix="LRB2-PILOT")

    # the dev tasks are held out of the high-tier pool at random, never more than a fifth of
    # it. the library cap is applied after that, so it holds for the test split itself
    high = cap_categories(
        [task for task in reuse_tasks if task["reuse_tier"] == "high"]
    )
    rng.shuffle(high)
    n_dev = min(DEV_COUNT, len(high) // 5)
    chosen = {
        "test": cap_libraries(high[n_dev:]),
        "dev": high[:n_dev],
        "low_value": spread_across_libraries(
            cap_categories(
                [task for task in reuse_tasks if task["reuse_tier"] == "low"]
            ),
            count=LOW_VALUE_COUNT,
        ),
        "control": cap_categories(control_tasks),
    }
    return number_tasks(chosen, rng, prefix="LRB2")


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
        help="write every agent-kept task, uncapped and unsplit, into artefacts/pilot",
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
    if not args.pilot:
        left_out = tiers["high"] - len(splits["test"]) - len(splits["dev"])
        print(f"{left_out} high-tier task(s) are left out by the variety caps")
    for name, rows in splits.items():
        if not rows:
            print(f"{name}: no tasks, so no file is written")
            continue
        save_jsonl(rows, out_dir / f"{name}.jsonl")
        print(f"{name}: {len(rows)} tasks -> {out_dir / name}.jsonl")


if __name__ == "__main__":
    main()
