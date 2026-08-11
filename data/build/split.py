"""Shared train/val/test splits for the dataset builders.

Each build script groups its rows by some key (category for the library-reuse splits, data
source for skywork), then hands them here. Grouping keeps the val/test holdouts balanced
across categories/sources rather than splitting everything together at random.

`stratified_split` is the simple balanced split (used by skywork and product_choice).
`split_disjoint_test` additionally holds out whole groups (e.g. tasks) for the test set, so the
library-usage test measures generalisation to unseen tasks while val stays in-distribution.
"""

import random
from collections import defaultdict
from collections.abc import Callable


def stratified_split(
    rows: list[dict],
    key: Callable[[dict], str],
    val_frac: float,
    seed: int = 42,
    floor_holdout: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into train/val/test, stratified by `key`.

    Each group is shuffled and contributes an equal-sized slice to val and test
    (`val_frac` of the group each), with the remainder going to train. With `floor_holdout`
    a group of at least three rows always yields at least one val/test row (so small but
    non-trivial categories are still represented in both holdouts); tiny groups then go
    entirely to train.

    Returns the (train, val, test) row lists, each shuffled.
    """
    rng = random.Random(seed)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)

    train, val, test = [], [], []
    for group_key in sorted(groups):
        group = groups[group_key][:]
        rng.shuffle(group)

        if floor_holdout:
            holdout = max(1, round(len(group) * val_frac)) if len(group) >= 3 else 0
        else:
            holdout = round(len(group) * val_frac)

        val += group[:holdout]
        test += group[holdout : holdout * 2]
        train += group[holdout * 2 :]

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def assign_task_ids(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    prefix: str,
    digits: int = 4,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stamp a unique `task_id` on every row, numbered across the splits.

    Ids are assigned after splitting, in test -> train -> val order, so they are contiguous
    within each split and the test set starts at `{prefix}0000`. The id becomes each row's
    first field.

    Returns the (train, val, test) splits with ids added.
    """
    counter = 0
    labelled: list[list[dict]] = []
    for split in (test, train, val):
        out = []
        for row in split:
            out.append({"task_id": f"{prefix}{counter:0{digits}d}", **row})
            counter += 1
        labelled.append(out)

    test_labelled, train_labelled, val_labelled = labelled
    return train_labelled, val_labelled, test_labelled


def split_disjoint_test(
    rows: list[dict],
    group_key: Callable[[dict], str],
    stratify_key: Callable[[dict], str],
    test_frac: float,
    val_frac: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows so test holds out whole groups (unseen), while val is an in-distribution subset.

    The test set is made group-disjoint: a selection of whole `group_key` groups (e.g. tasks) is
    removed from training entirely, spread across `stratify_key` categories with at most one group
    per category, so every category keeps at least one group for train/val. The val set is then a
    `stratify_key`-stratified holdout of rows from the remaining (seen) groups. So test measures
    generalisation to unseen tasks, while val measures learning on familiar ones.

    Returns the (train, val, test) row lists, each shuffled.
    """
    rng = random.Random(seed)

    # index rows by category -> group -> rows
    by_category: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_category[stratify_key(row)][group_key(row)].append(row)

    # choose the held-out test groups: whole groups, spread across categories (at most one each),
    # and only from categories with more than one group so train/val never lose a category
    n_groups = sum(len(groups) for groups in by_category.values())
    n_test = round(n_groups * test_frac)
    eligible = [cat for cat, groups in by_category.items() if len(groups) > 1]
    rng.shuffle(eligible)

    test_groups: set[tuple[str, str]] = set()
    for cat in eligible:
        if len(test_groups) >= n_test:
            break
        group_ids = sorted(by_category[cat])
        rng.shuffle(group_ids)
        test_groups.add((cat, group_ids[0]))

    # send test groups to test, everything else to the train/val pool
    test: list[dict] = []
    remaining: list[dict] = []
    for cat, groups in by_category.items():
        for group_id, group_rows in groups.items():
            target = test if (cat, group_id) in test_groups else remaining
            target.extend(group_rows)

    # val: a category-stratified row-level holdout of the seen (remaining) rows
    remaining_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in remaining:
        remaining_by_category[stratify_key(row)].append(row)

    train, val = [], []
    for cat in sorted(remaining_by_category):
        group = remaining_by_category[cat][:]
        rng.shuffle(group)
        holdout = max(1, round(len(group) * val_frac)) if len(group) >= 3 else 0
        val += group[:holdout]
        train += group[holdout:]

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test
