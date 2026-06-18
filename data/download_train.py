"""Download the code subset of Skywork/Skywork-OR1-RL-Data for RL training.

Source: https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data

Only the taco-sourced (stdin/stdout) problems are kept - the leetcode-sourced
problems force a fixed function signature, which constrains the model's
freedom over its own code structure and imports, so they're excluded.

Unlike the evaluation datasets in data/download_eval.py, this dataset is too large
to commit to the repo (~700MB after filtering), so the outputs are written to
data/train/, which is gitignored - this script must be re-run to regenerate
them locally or on a server.

Usage: python data/download_train.py
"""

import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from llm_cgr import load_jsonl, save_jsonl


# split outputs are written here - gitignored, since the dataset is too large to commit
OUTPUT_DIR = Path(__file__).parent / "train"

# commit hash pinned for reproducibility
REVISIONS = {
    "skywork_or1": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
}

# used to deterministically shuffle and split the data
SEED = 42

# train/val/test split ratio - val and test are kept equally sized
TRAIN_FRAC = 0.95
VAL_FRAC = 0.025
TEST_FRAC = 0.025


def download_skywork() -> Dataset:
    """Download the three code-subset parquet shards of skywork-or1-rl-data.

    Downloaded directly rather than via load_dataset so we only fetch the code
    shards and not the (separate) math subset.

    Returns the concatenated dataset of all three shards.
    """
    skywork_datasets = []
    for shard in range(3):
        parquet_path = hf_hub_download(
            repo_id="Skywork/Skywork-OR1-RL-Data",
            filename=f"data/code-0000{shard}-of-00003.parquet",
            repo_type="dataset",
            revision=REVISIONS["skywork_or1"],
        )
        skywork_datasets.append(Dataset.from_parquet(parquet_path))

    skywork = concatenate_datasets(skywork_datasets)
    print(f"loaded {len(skywork)} problems")
    print(f"  data sources: {Counter(skywork['data_source'])}")
    return skywork


def is_pure_stdin(
    row: dict,
) -> bool:
    """Check whether a skywork problem is solved via stdin/stdout only (no function call).

    Leetcode-sourced rows are always function-signature problems. A minority of
    taco-sourced rows are too - despite being grouped under "taco" in
    data_source, their ground_truth still sets fn_name and their prompt still
    asks for a fixed method (e.g. "class Solution: def add_all(lst):"), the same
    constraint leetcode rows have - so they're excluded here as well.

    Returns True if the row is taco-sourced and its ground truth has no fn_name set.
    """
    if "taco" not in row["data_source"]:
        return False

    ground_truth = json.loads(row["reward_model"]["ground_truth"])
    return not ground_truth.get("fn_name")


def build_records(
    skywork: Dataset,
) -> list[dict]:
    """Filter to pure stdin/stdout problems and format them into flat records.

    Each row's prompt is a single user message, and reward_model.ground_truth is
    a json string of {inputs, outputs} - the same format used by data/download_eval.py.

    Returns the filtered, formatted records.
    """
    records = [
        {
            "prompt": row["prompt"][0]["content"],
            "ground_truth": json.loads(row["reward_model"]["ground_truth"]),
            "data_source": row["data_source"],
            "source": "skywork_or1",
        }
        for row in skywork
        if is_pure_stdin(row)
    ]
    print(
        f"built {len(records)} records out of {len(skywork)} total "
        "(leetcode and function-call taco problems excluded)"
    )
    return records


def stratified_split(
    records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test, stratified by data_source.

    Each data_source group is shuffled and cut into equally-sized val/test
    chunks (VAL_FRAC/TEST_FRAC of the group) plus a train remainder, so the
    difficulty proportions - and the val/test sizes - are kept consistent
    across groups, rather than splitting all records together at random.

    Returns the (train, val, test) record lists.
    """
    rng = random.Random(SEED)

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["data_source"], []).append(record)

    train, val, test = [], [], []
    for source in sorted(groups):
        group = groups[source][:]
        rng.shuffle(group)

        # val and test take the same number of records from each group, so
        # their overall sizes come out exactly equal
        num_holdout = round(len(group) * VAL_FRAC)

        val += group[:num_holdout]
        test += group[num_holdout : num_holdout * 2]
        train += group[num_holdout * 2 :]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    """Download, filter, split, and save the skywork-or1 code training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skywork = download_skywork()
    records = build_records(skywork)

    train_records, val_records, test_records = stratified_split(records)
    print(
        f"split into train: {len(train_records)}, "
        f"val: {len(val_records)}, test: {len(test_records)}"
    )

    splits = {"train": train_records, "val": val_records, "test": test_records}
    for name, split_records in splits.items():
        save_jsonl(split_records, str(OUTPUT_DIR / f"{name}.jsonl"))
    print(f"saved train/val/test splits to {OUTPUT_DIR}/")

    print("\nsanity check:")
    for name in splits:
        split_records = load_jsonl(str(OUTPUT_DIR / f"{name}.jsonl"))
        sample = split_records[0]
        print(f"\n{name}: {len(split_records)} records")
        print(
            f"  data_source breakdown: {Counter(r['data_source'] for r in split_records)}"
        )
        print(f"  prompt (first 200 chars): {sample['prompt'][:200]!r}")


if __name__ == "__main__":
    main()
