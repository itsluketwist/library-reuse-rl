"""Load the project's training and evaluation datasets, building chat-message prompts."""

import logging
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict


logger = logging.getLogger(__name__)


def _build_messages(
    prompt: str,
    system_prompt: str | None,
) -> list[dict]:
    """Build the chat-messages list GRPOTrainer (and inference) expect from a raw prompt.

    Returns the messages, with the optional system prompt first.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _make_conversation(
    example: dict,
    system_prompt: str | None,
) -> dict:
    """Build the chat-messages "prompt" field from a raw skywork-format record.

    Also flattens `ground_truth` into the top-level `inputs`/`expected_outputs`
    fields that `src.response.evaluate.evaluate_record` reads.

    Returns the fields to add/overwrite on the example.
    """
    return {
        "prompt": _build_messages(example["prompt"], system_prompt),
        "inputs": example["ground_truth"]["inputs"],
        "expected_outputs": example["ground_truth"]["outputs"],
    }


def load_train_dataset(
    dataset_dir: str | Path,
    system_prompt: str | None,
    train_n: int | None = None,
    val_n: int | None = None,
) -> DatasetDict:
    """Load the skywork-format train/val splits from `dataset_dir` and build chat prompts.

    `dataset_dir` must contain train.jsonl and val.jsonl files in the schema produced
    by data/download_skywork.py: {prompt, ground_truth: {inputs, outputs}, data_source,
    source}.

    `train_n`/`val_n` truncate the train/val splits to at most that many rows (e.g. for
    a quick debug smoke test) - leave as None to use the full split.

    Returns a DatasetDict with "train" and "val" splits, each with the raw "prompt"
    column replaced by a chat-messages list, and "inputs"/"expected_outputs" columns
    added for the reward function to evaluate against.
    """
    dataset_dir = Path(dataset_dir)
    split_n = {"train": train_n, "val": val_n}
    splits = {}
    for split in ("train", "val"):
        split_file = dataset_dir / f"{split}.jsonl"
        logger.info(f"Loading {split} split from {split_file}")
        split_dataset = datasets.load_dataset("json", data_files=str(split_file))[
            "train"
        ]

        n = split_n[split]
        if n is not None:
            split_dataset = split_dataset.select(range(min(n, len(split_dataset))))
            logger.info(f"Truncated {split} split to {len(split_dataset)} rows")

        splits[split] = split_dataset

    dataset = DatasetDict(splits)
    dataset = dataset.map(
        lambda example: _make_conversation(example, system_prompt=system_prompt),
    )
    return dataset


def load_eval_dataset(
    dataset_file: str | Path,
    system_prompt: str | None,
    n: int | None = None,
) -> Dataset:
    """Load an evaluation dataset file and build chat prompts.

    Handles both the flat {prompt, inputs, expected_outputs, source} schema (from
    data/download_eval.py) and the nested ground_truth schema (from
    data/download_skywork.py), flattening the latter via _make_conversation.

    `n` truncates the dataset to at most that many rows (e.g. for a quick debug
    smoke test) - leave as None to use the full dataset.

    Returns the dataset with its raw "prompt" column replaced by a chat-messages
    list, and "inputs"/"expected_outputs" columns present at the top level.
    """
    dataset = datasets.load_dataset("json", data_files=str(dataset_file))["train"]

    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
        logger.info(f"Truncated dataset to {len(dataset)} rows")

    if "ground_truth" in dataset.column_names:
        dataset = dataset.map(
            lambda example: _make_conversation(example, system_prompt=system_prompt),
        )
    else:
        dataset = dataset.map(
            lambda example: {
                "prompt": _build_messages(example["prompt"], system_prompt)
            },
        )
    return dataset
